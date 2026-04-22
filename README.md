# miniLLM

这是一个从 0 开始实现的极简 GPT 语言模型。它的目标不是训练出很强的大模型，而是把 LLM 最核心的一条链路完整跑通：

```text
文本语料 -> tokenizer -> token ids -> batch -> Transformer -> loss -> 反向传播 -> checkpoint -> 自回归生成
```

当前版本参考 nanoGPT 的思路，实现了一个 **char-level decoder-only Transformer**。char-level tokenizer 很简单，但好处是透明：你可以直接看到字符如何变成整数，整数如何进入模型，模型又如何预测下一个字符。

## 文件结构

```text
miniLLM/
├── data/sample.txt          # 一个很小的示例训练语料
├── minillm/
│   ├── data.py              # 读取文本、切分 train/val、构造 next-token batch
│   ├── model.py             # GPTConfig、CausalSelfAttention、Block、MiniGPT
│   └── tokenizer.py         # char-level tokenizer
├── train.py                 # 训练入口
├── generate.py              # 文本生成入口
├── requirements.txt
└── README.md
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

用默认小语料训练一个玩具模型：

```bash
python train.py --max_steps 300 --block_size 64 --batch_size 16
```

生成文本：

```bash
python generate.py --prompt "To build" --max_new_tokens 200
```

如果你在 Apple Silicon 上，默认会自动尝试使用 `mps`；有 NVIDIA GPU 时会自动使用 `cuda`；否则使用 `cpu`。

## 核心概念

### 1. Tokenizer

文件：`minillm/tokenizer.py`

模型不能直接处理字符串，所以 tokenizer 负责：

```text
"hello" -> [token_id_1, token_id_2, ...]
```

当前实现是字符级 tokenizer：

- 统计训练语料中出现过的所有字符
- 给每个字符分配一个整数 id
- 训练时把文本变成 token id 序列
- 生成后把 token id 还原成字符串

真实 LLM 常用 BPE、SentencePiece、Unigram 等 tokenizer，因为它们能把文本压缩得更短，训练和推理更省。但 char-level 是理解流程的最好起点。

### 2. Next-token Prediction

文件：`minillm/data.py`

语言模型训练的核心任务是预测下一个 token：

```text
输入 x:  [t0, t1, t2, t3]
目标 y:  [t1, t2, t3, t4]
```

也就是说，模型在每个位置都要猜“下一个 token 是什么”。训练损失使用 cross entropy。

### 3. Decoder-only Transformer

文件：`minillm/model.py`

模型结构大致是：

```text
token ids
  -> token embedding
  -> position embedding
  -> N 个 Transformer Block
  -> LayerNorm
  -> LM Head
  -> logits
```

每个 Transformer Block 包含：

```text
x = x + causal_self_attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

这里使用的是 pre-norm 结构，训练小模型时更稳定，也更接近现代 GPT/LLaMA 系模型的常见写法。

### 4. Causal Self-Attention

普通 self-attention 会让每个位置看到所有 token。但语言模型训练时，位置 `i` 不能看到未来的 token，否则它就作弊了。

所以我们使用 causal mask：

```text
位置 0 只能看 0
位置 1 可以看 0, 1
位置 2 可以看 0, 1, 2
...
```

这就是 decoder-only GPT 能进行自回归生成的关键。

### 5. 自回归生成

文件：`generate.py`

生成时模型一次只预测一个新 token：

```text
prompt -> 预测 next token -> 拼回输入 -> 再预测下一个 token -> ...
```

`temperature` 控制随机性：

- 低 temperature：更保守，更容易重复
- 高 temperature：更随机，更发散

`top_k` 只允许模型从概率最高的 k 个 token 中采样，可以减少很离谱的输出。

## 训练产物

训练后会生成：

```text
out/
├── tokenizer.json    # tokenizer 词表
├── model.pt          # 模型 checkpoint
└── metadata.json     # 配置和最后一次 loss
```

`model.pt` 保存了：

- 模型参数
- 模型配置
- 训练 step
- 最后一次 train/val loss

## 你可以怎么继续扩展

建议按这个顺序升级：

1. 把 char tokenizer 换成 BPE tokenizer
2. 把 learned position embedding 换成 RoPE
3. 把 LayerNorm 换成 RMSNorm
4. 把 GELU MLP 换成 SwiGLU
5. 加入 KV cache，让生成更快
6. 加入 gradient accumulation，模拟更大 batch size
7. 加入 mixed precision，学习 fp16/bf16 训练
8. 加入 LoRA，做参数高效微调
9. 加入简单 eval benchmark，比如困惑度、QA 准确率
10. 写一个 C 或 C++ 推理器，理解权重导出和部署

这条路线会从 nanoGPT 逐渐走向 LLaMA-style mini model，也会覆盖很多面试里常问的 LLM 工程点。

