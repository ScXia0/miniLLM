"""一个紧凑的 decoder-only Transformer，支持 GPT-style 和 LLaMA-style 配置。

这个文件刻意把模型结构写得比较直白，不把复杂度藏在框架后面。你可以在这里
直接看到 GPT/LLaMA 类模型常见的组成部分：token embedding、位置编码、
causal attention、MLP、残差连接、归一化层，以及 next-token loss。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """模型大小、上下文长度和结构风格的超参数。"""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    position_embedding_type: str = "learned"
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """RMSNorm，Root Mean Square Normalization，LLaMA 风格模型常用的归一化层。

    LayerNorm 会先减均值，再除以标准差；RMSNorm 不减均值，只用 root mean
    square，也就是均方根做归一化。它结构更简单、计算略省，在 LLaMA 等现代
    decoder-only LLM 中很常见。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 为了数值稳定，归一化统计量用 fp32 计算；最后再转回输入 dtype，
        # 这样 mixed precision 仍然能节省显存。
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight.to(dtype=x.dtype) * normed.to(dtype=x.dtype)


def build_norm(config: GPTConfig) -> nn.Module:
    """根据配置创建归一化层。"""

    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError(f"unknown norm_type: {config.norm_type}")


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    theta: float,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 应用到 query 和 key 上。

    RoPE 是 Rotary Position Embedding，中文常叫“旋转位置编码”。它会按 token
    位置给 query/key 的成对维度做旋转。和 learned absolute position embedding
    不同，RoPE 不是把位置向量直接加到 token embedding 上，而是把位置信息注入
    attention 的 q/k 计算里，因此在 LLaMA 风格架构中很常见。

    position_offset 表示当前 token 在整段上下文中的起始位置。
    这在 KV cache 推理时很重要：decode 阶段一次只输入一个新 token，但它的
    实际位置不再是 0，而是“历史长度”。

    输入形状：
        q, k: (B, n_head, T, head_dim)
    """

    seq_len = q.size(-2)
    head_dim = q.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")

    positions = torch.arange(position_offset, position_offset + seq_len, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype=q.dtype)[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return x_rotated.flatten(start_dim=-2)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    """带 causal mask 的多头自注意力。

    causal 的意思是：第 i 个 token 只能看见位置 <= i 的 token，不能偷看未来。
    这是 decoder-only Transformer 可以做自回归生成的关键限制。训练时我们仍然
    能并行计算整段序列，但 causal mask 会阻止未来信息泄漏。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.position_embedding_type == "rope"
        self.rope_theta = config.rope_theta
        if self.use_rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires n_embd / n_head to be even")

        # 为了效率，用一个线性层同时生成 q/k/v。
        # q = query 查询向量，k = key 键向量，v = value 值向量。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 下三角 causal mask 存成 buffer：它会跟着模型移动到对应 device，
        # 但不是可训练参数，不会被优化器更新。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """执行注意力计算，并在需要时返回新的 KV cache。

        past_kv:
            历史 key/value 缓存，形状分别是 (B, n_head, T_past, head_dim)。

        use_cache:
            是否返回当前层新的 cache。训练时为 False；生成时通常为 True。

        position_offset:
            当前输入 token 在整段上下文中的起始位置。对 RoPE 和 learned position
            embedding 都很重要，因为 decode 阶段输入很短，但位置并不从 0 开始。
        """

        batch, seq_len, embd = x.shape

        q, k, v = self.c_attn(x).split(embd, dim=2)

        # 形状约定：
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        # B=batch size，T=序列长度，C=embedding 维度。
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, k, self.rope_theta, position_offset=position_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v) if use_cache else None
        total_len = k.size(-2)

        # attention score 是 query 和 key 的缩放点积。
        # 除以 sqrt(head_dim) 可以避免 logits 过大，让 softmax 更稳定。
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)

        # 训练时通常没有 past_kv，直接取左上角的标准下三角 mask。
        # 使用 KV cache 时，当前 q 对应的是“靠后的位置”，所以要从大 mask 中
        # 取出 [position_offset : position_offset + seq_len] 这一段行。
        mask = self.causal_mask[:, :, position_offset : position_offset + seq_len, :total_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 用注意力权重对 value 做加权求和，然后把多个 head 合并回一个 embedding。
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        out = self.resid_dropout(self.c_proj(out))
        return out, present_kv


class MLP(nn.Module):
    """Transformer block 里的前馈网络，也就是 attention 后面的 MLP。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type
        if config.mlp_type == "gelu":
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout),
            )
        elif config.mlp_type == "swiglu":
            # SwiGLU 是 LLaMA 常用的 gated MLP（门控前馈网络）。
            # 它大致计算：down_proj(silu(gate_proj(x)) * up_proj(x))。
            # 这里的 gate 像一个“开关”：先用 SiLU 激活生成门控信号，再和另一条
            # up projection 分支逐元素相乘。相比普通 GELU MLP，这种乘法交互在
            # LLM 规模下通常表达能力更强。
            hidden_dim = 4 * config.n_embd
            self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
            self.dropout = nn.Dropout(config.dropout)
        else:
            raise ValueError(f"unknown mlp_type: {config.mlp_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            return self.net(x)
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class Block(nn.Module):
    """一个 pre-norm Transformer block。

    残差结构是：
        x = x + attention(layer_norm(x))
        x = x + mlp(layer_norm(x))

    pre-norm 指先归一化再进入 attention/MLP。相比更早的 post-norm，pre-norm
    在模型变深时通常更稳定，因此现代 decoder-only LLM 中很常见。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present_kv = self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class MiniGPT(nn.Module):
    """一个小型 decoder-only 语言模型，在每个位置预测下一个 token。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.position_embedding_type == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        elif config.position_embedding_type == "rope":
            self.position_embedding = None
        else:
            raise ValueError(f"unknown position_embedding_type: {config.position_embedding_type}")

        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying（权重绑定）：输入 token embedding 和输出分类头共享权重。
        # 这样既减少参数量，也常常能提升小型语言模型的效果。
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """使用 GPT 风格的小标准差正态分布初始化权重。"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_impl(
        self,
        idx: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        """模型内部前向逻辑，供训练和 KV cache 推理共用。

        训练时：
            past_kv=None, use_cache=False

        推理时：
            prefill 阶段把整段 prompt 输入进来，建立每一层的 cache
            decode 阶段每次只输入新 token，并把 cache 传回来复用
        """

        batch, seq_len = idx.shape
        if past_kv is None:
            past_kv = [None] * len(self.blocks)
        if len(past_kv) != len(self.blocks):
            raise ValueError("past_kv length must match number of blocks")

        past_length = 0
        if past_kv and past_kv[0] is not None:
            past_length = past_kv[0][0].size(-2)

        if past_length + seq_len > self.config.block_size:
            raise ValueError(
                f"context length {past_length + seq_len} exceeds block_size {self.config.block_size}"
            )

        tok_emb = self.token_embedding(idx)
        if self.position_embedding is None:
            x = self.dropout(tok_emb)
        else:
            positions = torch.arange(
                past_length,
                past_length + seq_len,
                dtype=torch.long,
                device=idx.device,
            )
            pos_emb = self.position_embedding(positions)
            x = self.dropout(tok_emb + pos_emb)

        present_kv = [] if use_cache else None
        for block, layer_past in zip(self.blocks, past_kv):
            x, layer_present = block(
                x,
                past_kv=layer_past,
                use_cache=use_cache,
                position_offset=past_length,
            )
            if use_cache:
                assert present_kv is not None
                present_kv.append(layer_present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, present_kv

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """执行一次前向传播。

        参数：
            idx: 输入 token id，形状为 (B, T)
            targets: 可选的目标 token id，形状为 (B, T)

        返回：
            logits: 未归一化的预测分数，形状为 (B, T, vocab_size)
            loss: 如果传入 targets，则返回 cross entropy loss，否则为 None
        """

        logits, _ = self._forward_impl(idx, past_kv=None, use_cache=False)

        loss = None
        if targets is not None:
            # Cross entropy 期望输入形状是 (N, C)，所以把 B*T 个位置展平成 N。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def forward_with_kv_cache(
        self,
        idx: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        """推理专用前向：返回 logits 和更新后的 KV cache。

        KV cache 的直觉是：
        - K = key
        - V = value
        - 旧 token 的 k/v 在后续 decode 时不会变，所以可以缓存下来
        - 新 token 来时，只算它自己的 q/k/v，然后直接和历史 cache 拼起来用

        这样就避免了“每生成 1 个 token，就把整个上下文再算一遍”的重复工作。
        """

        logits, present_kv = self._forward_impl(idx, past_kv=past_kv, use_cache=True)
        assert present_kv is not None
        return logits, present_kv

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成：每次预测一个新 token，并把它拼回上下文继续预测。

        use_kv_cache=False:
            每一步都重算当前整段上下文，代码最直观，但重复计算很多。

        use_kv_cache=True:
            prompt 先做一次 prefill，建立 cache；之后 decode 阶段每次只输入一个
            新 token，并复用历史 key/value。
        """

        self.eval()
        if not use_kv_cache:
            for _ in range(max_new_tokens):
                # 模型最多只能看 block_size 个上下文 token。prompt 很长时，只保留
                # 最近的上下文窗口，这也是 GPT 类模型推理时常见的处理方式。
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

        # prefill：先把整段 prompt 走一遍，得到最后一个位置的 logits 和每一层缓存。
        idx_cond = idx[:, -self.config.block_size :]
        idx = idx_cond
        logits, past_kv = self.forward_with_kv_cache(idx_cond)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            # 如果 cache 还没装满 block_size，decode 阶段只需要输入一个新 token。
            # 如果已经达到窗口上限，就按“最近 block_size 个 token”重建一次 cache。
            # 这样可以保持与无 cache 路径相同的最近窗口语义。
            cache_len = past_kv[0][0].size(-2) if past_kv and past_kv[0] is not None else 0
            if cache_len >= self.config.block_size:
                idx_window = idx[:, -self.config.block_size :]
                logits, past_kv = self.forward_with_kv_cache(idx_window)
            else:
                logits, past_kv = self.forward_with_kv_cache(next_id, past_kv=past_kv)

        return idx

    def num_parameters(self) -> int:
        """返回可训练参数数量。"""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
