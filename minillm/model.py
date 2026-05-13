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
class ModelConfig:
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
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "attn"


class LoRALinear(nn.Module):
    """LoRA 版线性层：冻结原线性层，只额外训练一个低秩增量。

    普通线性层本来是：

        y = xW

    LoRA 的直觉是不要直接改整个大矩阵 W，而是冻结 W，只额外学习一个小增量：

        y = x(W + ΔW)
        ΔW = A @ B

    其中：
    - A 的形状大致是 (in_features, r)
    - B 的形状大致是 (r, out_features)
    - r 很小，所以新增参数量远小于整个 W

    一个最小例子：

        假设原权重 W.shape = (4096, 4096)

        全量微调时，需要更新约 1677 万个参数。

        如果 LoRA rank = 8，则：
            A.shape = (4096, 8)
            B.shape = (8, 4096)

        需要训练的参数量变成：
            4096*8 + 8*4096 = 65536

    也就是说，LoRA 不是把原线性层删掉换成一个小层，而是：
    - 保留原始大线性层 base
    - 旁边再挂一条低秩增量分支
    - 前向时输出 = base_out + lora_out

    所以 LoRA 真正减少的不是“这一层的输出维度”，而是：
    - 需要训练的参数量
    - 参数更新的自由度

    例如：
    - 原层还是输出 384 维
    - LoRA 分支最后也必须输出 384 维，才能和原层逐元素相加
    - 低秩发生在中间那个很小的 rank 维度上，比如 8
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA 分支的形状：
        # x: (..., in_features)
        # A: in_features -> rank
        # B: rank -> out_features
        #
        # 用一个特别具体的小例子看：
        #
        #     假设原线性层是：
        #         base = Linear(128, 384)
        #
        #     这通常意味着：
        #         x.shape              = (B, T, 128)
        #         base(x).shape        = (B, T, 384)
        #
        #     如果 LoRA rank = 8，则：
        #         lora_A.weight.shape  = (8, 128)
        #         lora_B.weight.shape  = (384, 8)
        #
        #     前向时会发生：
        #         x                  -> lora_A -> (B, T, 8)
        #         (B, T, 8)          -> lora_B -> (B, T, 384)
        #
        #     最后把这个 LoRA 增量加回 base(x)：
        #         output = base(x) + lora_B(lora_A(x)) * scaling
        #
        # 这里要特别注意一个很容易混淆的点：
        # - LoRA 不是把原来的 Linear(128, 384) 变成更小的 Linear(128, 8)
        # - 而是保留原来的 Linear(128, 384)
        # - 再额外挂一条 128 -> 8 -> 384 的修正支路
        #
        # 所以从层接口看，外面仍然只会看到：
        #     输入 128 维 -> 输出 384 维
        #
        # 少掉的是“可训练参数量”，不是“输出通道数”。
        #
        # 注意：
        # PyTorch 的 nn.Linear 存权重时是 (out_features, in_features)，
        # 所以上面看到的 lora_A / lora_B.weight 形状，
        # 和我们讲原理时写的 A:(in, r)、B:(r, out) 是同一件事，只是转置视角不同。
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        # 原始大权重冻结，LoRA 小矩阵可训练。
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def init_lora_parameters(self) -> None:
        """初始化 LoRA 小矩阵。

        常见做法是：
        - A 随机初始化
        - B 初始化为 0

        这样一开始 ΔW = A @ 0 = 0，模型起点与原始 base model 完全一致。
        """

        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里把 LoRA 前向完全展开看一遍。
        #
        # 先设一个统一例子：
        #
        #     base = Linear(128, 384)
        #     rank = 8
        #
        # 并假设输入：
        #
        #     x.shape = (2, 16, 128)
        #
        # 这 3 个维度分别表示：
        # - 2   = batch size，表示一次送进来 2 条样本
        # - 16  = seq_len，表示每条样本有 16 个 token
        # - 128 = in_features，也就是“每个 token 的隐藏向量长度”
        #
        # ------------------------------------------------------------
        # 第 1 步：原始线性层输出
        # ------------------------------------------------------------
        #
        #     base_out = self.base(x)
        #
        # 因为 self.base 是 Linear(128, 384)，它做的事是：
        # - 输入最后一维必须是 128
        # - 输出最后一维会变成 384
        #
        # 所以：
        #
        #     x.shape        = (2, 16, 128)
        #     base_out.shape = (2, 16, 384)
        #
        # 为什么前两维 2 和 16 不变？
        # 因为 nn.Linear 只作用在“最后一维特征维”上：
        # - 不会改变 batch size
        # - 不会改变序列长度
        # - 只会把每个 token 对应的 128 维向量映射成 384 维向量
        #
        # ------------------------------------------------------------
        # 第 2 步：LoRA 分支输入 dropout
        # ------------------------------------------------------------
        #
        #     dropped_x = self.lora_dropout(x)
        #
        # dropout 不会改张量形状，只会随机把一部分值置 0。
        #
        # 所以：
        #
        #     dropped_x.shape = (2, 16, 128)
        #
        # ------------------------------------------------------------
        # 第 3 步：先过 lora_A，把大特征压到低秩空间
        # ------------------------------------------------------------
        #
        #     a_out = self.lora_A(dropped_x)
        #
        # lora_A 是 Linear(128, 8)，表示：
        # - 输入最后一维 128
        # - 输出最后一维 8
        #
        # 所以：
        #
        #     dropped_x.shape = (2, 16, 128)
        #     a_out.shape     = (2, 16, 8)
        #
        # 这一步可以理解成：
        # “先把每个 token 的 128 维表示，压缩到一个很小的 rank=8 空间里”
        #
        # ------------------------------------------------------------
        # 第 4 步：再过 lora_B，从低秩空间投回目标维度
        # ------------------------------------------------------------
        #
        #     b_out = self.lora_B(a_out)
        #
        # lora_B 是 Linear(8, 384)，表示：
        # - 输入最后一维 8
        # - 输出最后一维 384
        #
        # 所以：
        #
        #     a_out.shape     = (2, 16, 8)
        #     b_out.shape     = (2, 16, 384)
        #
        # 到这里，LoRA 分支已经生成了一个“增量输出”，它的形状必须和 base_out 一样，
        # 因为后面两者要逐元素相加。
        #
        # ------------------------------------------------------------
        # 第 5 步：乘缩放系数 scaling
        # ------------------------------------------------------------
        #
        #     lora_out = b_out * self.scaling
        #
        # self.scaling = alpha / rank，是一个标量。
        # 标量乘张量不会改形状，只会改数值大小。
        #
        # 所以：
        #
        #     lora_out.shape = (2, 16, 384)
        #
        # ------------------------------------------------------------
        # 第 6 步：和原始输出相加
        # ------------------------------------------------------------
        #
        #     output = base_out + lora_out
        #
        # 因为：
        #
        #     base_out.shape = (2, 16, 384)
        #     lora_out.shape = (2, 16, 384)
        #
        # 所以可以逐元素相加，得到：
        #
        #     output.shape = (2, 16, 384)
        #
        # 这就是为什么 LoRA 不会改变这个线性层的“外部接口”：
        # - 输入仍然是 (..., 128)
        # - 输出仍然是 (..., 384)
        #
        # 它只是偷偷在内部多算了一条低秩分支，然后把这条分支加回原输出。
        #
        # 用公式写就是：
        #
        #     base_out = xW
        #     lora_out = xAB * (alpha / rank)
        #     output   = xW + xAB * (alpha / rank)
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


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


def build_norm(config: ModelConfig) -> nn.Module:
    """根据配置创建归一化层。"""

    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError(f"unknown norm_type: {config.norm_type}")


def should_apply_lora(config: ModelConfig, module_group: str) -> bool:
    """判断某个模块分组是否应该注入 LoRA。

    这里用最简单的三档：

    - "attn"：只给 attention 里的线性层加 LoRA
    - "mlp"：只给 MLP 里的线性层加 LoRA
    - "all"：attention + MLP 都加

    注意这里控制的是：
    - 哪些线性层会被“LoRA 化”

    不是控制：
    - 前向时走不走这些层
    - 训练时是否跳过这些层

    只要某层被包成 LoRALinear，前向时仍然会同时经过：
    - base(x)
    - lora_B(lora_A(x))

    最后做相加。
    真正“哪些参数会更新”是后面 freeze_non_lora_parameters() 决定的。
    """

    # 先看 rank。
    # rank <= 0 可以理解成“完全不开 LoRA”，于是所有线性层都保持普通 nn.Linear。
    if config.lora_rank <= 0:
        return False
    if config.lora_target_modules == "all":
        return module_group in {"attn", "mlp"}
    return config.lora_target_modules == module_group


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

    def __init__(self, config: ModelConfig) -> None:
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
        if should_apply_lora(config, "attn"):
            # attention 里这两个位置是 LoRA 最常见的注入点。
            # 这是因为 attention 里的投影层通常参数很多，而且对模型行为影响很大，
            # 所以“只改很少参数也能有效适配任务”的性价比往往比较高。
            #
            # 统一示例：
            # 假设 n_embd = 128。
            #
            # 则原本：
            #     c_attn = Linear(128, 384)   # 一次产出 q/k/v
            #     c_proj = Linear(128, 128)   # 多头拼回后再投影
            #
            # 开 LoRA 且 target_modules="attn" 后：
            #     self.c_attn 不再是普通 Linear，而是 LoRALinear(Linear(128, 384))
            #     self.c_proj 不再是普通 Linear，而是 LoRALinear(Linear(128, 128))
            #
            # 也就是说，attention 结构没变，输入输出形状也没变。
            # 变的是：
            # - 这两个线性层内部多了一条“低秩增量分支”
            # - 后续如果再调用 freeze_non_lora_parameters()，
            #   训练时就只更新这条增量分支里的 A/B 小矩阵
            #
            # 所以这里减少的不是 c_attn / c_proj 的输出维度，
            # 而是它们的“可训练参数量”。
            self.c_attn = LoRALinear(self.c_attn, config.lora_rank, config.lora_alpha, config.lora_dropout)
            self.c_proj = LoRALinear(self.c_proj, config.lora_rank, config.lora_alpha, config.lora_dropout)
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

        # KV cache：新 token 来了，只算它自己的 k/v，再和历史缓存的 past_k/past_v 拼起来
        # 并非每次把整个上下文过一遍，而是 旧的 K/V 直接复用 + 只补充新的KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        '''
            如果当前是 cache 模式
            就把这次更新后的 k/v 返回出去
            给下一步 decode 继续用
            所以这一层 forward 不再只返回 attention 输出，还会返回：“这一层最新的 cache”
        '''
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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.mlp_type = config.mlp_type
        if config.mlp_type == "gelu":
            fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            proj = nn.Linear(4 * config.n_embd, config.n_embd)
            if should_apply_lora(config, "mlp"):
                # 如果 target_modules="mlp"，LoRA 就不加在 attention 上，
                # 而是加在前馈网络这两个线性层上。
                #
                # 例如 n_embd = 128 时：
                #     fc   : Linear(128, 512)
                #     proj : Linear(512, 128)
                #
                # 这时训练的不是整个 fc/proj 大矩阵，
                # 而是它们各自额外挂上的低秩 A/B 小矩阵。
                fc = LoRALinear(fc, config.lora_rank, config.lora_alpha, config.lora_dropout)
                proj = LoRALinear(proj, config.lora_rank, config.lora_alpha, config.lora_dropout)
            self.net = nn.Sequential(
                fc,
                nn.GELU(),
                proj,
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
            if should_apply_lora(config, "mlp"):
                # SwiGLU 版 MLP 有三层线性投影：
                # gate_proj / up_proj / down_proj
                #
                # 所以这里如果给 MLP 打 LoRA，会同时包这三层。
                # 这样做的含义是：
                # attention 部分不动，而把“特征变换能力”的可学习增量主要放在 MLP 里。
                self.gate_proj = LoRALinear(
                    self.gate_proj, config.lora_rank, config.lora_alpha, config.lora_dropout
                )
                self.up_proj = LoRALinear(
                    self.up_proj, config.lora_rank, config.lora_alpha, config.lora_dropout
                )
                self.down_proj = LoRALinear(
                    self.down_proj, config.lora_rank, config.lora_alpha, config.lora_dropout
                )
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

    def __init__(self, config: ModelConfig) -> None:
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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd) # 把离散的 token id，映射成连续向量，长度为n_embd
        if config.position_embedding_type == "learned": # 显式
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        elif config.position_embedding_type == "rope": # 隐式：不单独构建位置 embedding 表，而是在 attention 里给 q/k 加位置信息：GPT-style：learned positional embedding；LLaMA-style：RoPE
            self.position_embedding = None
        else:
            raise ValueError(f"unknown position_embedding_type: {config.position_embedding_type}")

        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]) # Transformer block 堆很多层
        self.ln_f = build_norm(config) # 归一化
        # 把每个位置的隐藏向量，投影成“词表上每个 token 的分数”
        '''
            输入：长度为 n_embd 的隐藏向量
            输出：长度为 vocab_size 的分数向量
        '''
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 

        # Weight tying（权重绑定）：输入 token embedding 和输出分类头共享权重。
        # 这样既减少参数量，也常常能提升小型语言模型的效果。
        # 整个输入输出参数流程：model_io_example.txt
        '''
        LLM 里很经典的技巧:
            输入 embedding 用的权重矩阵
            输出预测头用的权重矩阵
            直接共享
        '''
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        self._init_lora_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        """使用 GPT 风格的小标准差正态分布初始化权重。"""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_lora_parameters(self) -> None:
        """单独初始化 LoRA 小矩阵。

        为什么要单独做：
        - `self.apply(self._init_weights)` 会递归初始化所有 nn.Linear
        - LoRA 的 lora_B 我们希望显式初始化成 0，而不是普通线性层的随机值

        这样做的效果是：
            一开始 LoRA 分支输出 0，模型行为与原始 base model 完全一致
        """

        # 统一示例：
        # 如果 blocks.0.attn.c_attn 被包成了 LoRALinear，
        # 那它里面已经有：
        # - base      : 原始大矩阵（继承 base checkpoint 或随机初始化）
        # - lora_A    : 小矩阵，随机初始化
        # - lora_B    : 小矩阵，清零初始化
        #
        # 这样训练第 0 步时：
        #     lora_out = 0
        #     layer_out = base_out + 0
        #
        # 所以模型起跑线和原始模型完全对齐。
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.init_lora_parameters()

    def freeze_non_lora_parameters(self) -> None:
        """冻结除 LoRA 之外的所有参数。

        这是 LoRA 训练最经典的设置：
        - base model 权重全部冻结
        - 只训练 lora_A / lora_B

        统一示例：
        假设某个 attention 线性层原本是：

            y = xW

        开启 LoRA 后会变成：

            y = xW + xAB

        此时：
        - W 冻结
        - 只更新 A 和 B

        这里也顺手澄清一个常见误区：
        - 开了 LoRA 以后，模型总参数量通常不会变少，甚至会略微增加
          因为原来的 W 还在，只是被冻结了；另外又新增了 A 和 B
        - 真正大幅减少的是“可训练参数量”
        - 所以前向仍然会经过完整模型，base 权重也仍然参与计算
          只是反向更新时不再改 base，只改 LoRA 小矩阵
        """

        # 这一步会把“训练哪些参数”改成非常鲜明的两段：
        #
        # 1. 先把整个模型全部冻住
        # 2. 再只把每个 LoRALinear 里的 lora_A / lora_B 解冻
        #
        # 于是优化器最终看到的 trainable params，大致就只剩：
        #     blocks.*.attn.c_attn.lora_A.weight
        #     blocks.*.attn.c_attn.lora_B.weight
        #     blocks.*.attn.c_proj.lora_A.weight
        #     blocks.*.attn.c_proj.lora_B.weight
        # 或者如果 target_modules="mlp"，则会对应到 MLP 内部那些 LoRA 权重。
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True

    def has_lora(self) -> bool:
        """判断当前模型里是否已经注入 LoRA。"""

        return any(isinstance(module, LoRALinear) for module in self.modules())

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
        '''
            每一层 block 都有自己的一份 past_kv
            forward 后每层都会产出自己新的 present_kv
            最后整合成整个模型的 cache 列表
            所以 KV cache 不是“模型只有一个缓存”，而是：

            每一层 Transformer block 都维护自己的 K/V 缓存
        '''
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
            #
            # ignore_index=-100 对 SFT 很重要：prompt 那些“不参与监督”的位置，
            # 可以直接把 label 写成 -100，cross entropy 会自动跳过。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

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
    # 自回归生成
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
        '''
        KV cache 推理通常分两段：
            1. prefill
            输入是：
            整段 prompt
            目的：
            计算整段 prompt 的前向传播
            为每一层建立历史 K/V
            得到 prompt 最后一个位置对应的 logits
            2. decode
            输入是：
            每次只输入 1 个新 token
            目的：
            复用 prefill 留下来的 past_kv
            只增量计算新 token 的结果
            继续往后生成
            所以你可以理解成：

            prefill：先把历史建立起来
            decode：之后基于这个历史一点点往后接
        '''
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

    def num_parameters(self, trainable_only: bool = True) -> int:
        """返回参数数量。

        默认返回“可训练参数数量”，因为在 LoRA 训练里这个数字最有意义。
        如果 trainable_only=False，则返回模型总参数量。
        """

        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
