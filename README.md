# miniLLM

这是一个从 0 开始实现的极简 GPT 语言模型。它的目标不是训练出很强的大模型，而是把 LLM 最核心的一条链路完整跑通：

```text
文本语料 -> tokenizer -> token ids -> batch -> Transformer -> loss -> 反向传播 -> checkpoint -> 自回归生成
```

当前版本参考 nanoGPT 的思路，实现了一个 **char-level decoder-only Transformer**。模型支持两种架构配置：

- `--architecture gpt`：GPT-2 风格，LayerNorm + GELU MLP + learned position embedding
- `--architecture llama`：LLaMA 风格，RMSNorm + SwiGLU + RoPE

char-level tokenizer 很简单，但好处是透明：你可以直接看到字符如何变成整数，整数如何进入模型，模型又如何预测下一个字符。

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
python train.py --architecture gpt --max_steps 300 --block_size 64 --batch_size 16
```

训练一个 LLaMA-style 玩具模型：

```bash
python train.py --architecture llama --max_steps 300 --block_size 64 --batch_size 16
```

生成文本：

```bash
python generate.py --prompt "To build" --max_new_tokens 200
```

如果你在 Apple Silicon 上，默认会自动尝试使用 `mps`；有 NVIDIA GPU 时会自动使用 `cuda`；否则使用 `cpu`。

## 从 0 跑通完整流程

下面这些命令可以按顺序执行。每一步都对应 LLM 训练/推理链路里的一个环节。

### 1. 获取代码

如果你是在自己的电脑上重新拉取项目：

```bash
git clone git@github.com:ScXia0/miniLLM.git
cd miniLLM
```

如果你已经在本机当前项目目录里：

```bash
cd /Users/didi/Documents/miniLLM
```

### 2. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

这个项目目前只依赖：

- `torch`：模型、训练、自动求导
- `numpy`：后续扩展数据处理时会用到

### 3. 做一次最小 smoke test

这一步只训练 1 step，用来确认环境、数据、模型、反向传播、checkpoint 保存都能跑通：

```bash
python3 train.py \
  --max_steps 1 \
  --eval_interval 1 \
  --eval_iters 1 \
  --out_dir out/smoke
```

正常情况下你会看到类似输出：

```text
device: cpu
vocab_size: 100
train tokens: ...
parameters: ...
step     1/1 | train ... | val ... | lr ... | ...
saved checkpoint to out/smoke/model.pt
```

这里的 loss 不重要，因为只训练了一步。重要的是整条链路没有报错。

### 4. 用 smoke checkpoint 生成文本

```bash
python3 generate.py \
  --checkpoint out/smoke/model.pt \
  --tokenizer out/smoke/tokenizer.json \
  --prompt "To build" \
  --max_new_tokens 100
```

只训练 1 step 时，输出大概率是随机字符。这是正常的，因为模型还没学到足够模式。

### 5. 训练一个稍微能看出模式的玩具模型

```bash
python3 train.py \
  --architecture gpt \
  --max_steps 300 \
  --block_size 64 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --out_dir out/demo
```

参数含义：

- `--max_steps`：训练多少步
- `--architecture`：模型结构，`gpt` 或 `llama`
- `--block_size`：上下文长度，也就是每次看多少个 token
- `--batch_size`：每个 microbatch 训练多少段文本
- `--gradient_accumulation_steps`：累积多少个 microbatch 后再更新一次参数
- `--n_layer`：Transformer block 层数
- `--n_head`：attention head 数量
- `--n_embd`：每个 token 的向量维度
- `--out_dir`：checkpoint 和 tokenizer 保存目录

有效 batch token 数可以这样理解：

```text
effective_batch_tokens = batch_size * block_size * gradient_accumulation_steps
```

这是真实 LLM 训练里很重要的概念：显存不够时，不一定要直接增大 `batch_size`，也可以用梯度累积模拟更大的 batch。

### 6. 用训练好的 demo 模型生成

```bash
python3 generate.py \
  --checkpoint out/demo/model.pt \
  --tokenizer out/demo/tokenizer.json \
  --prompt "To build" \
  --max_new_tokens 300 \
  --temperature 0.8 \
  --top_k 20
```

生成参数含义：

- `--prompt`：起始文本
- `--max_new_tokens`：最多生成多少个新 token
- `--temperature`：采样随机性，越低越保守，越高越发散
- `--top_k`：只从概率最高的 k 个 token 里采样

### 7. 换成你自己的训练语料

准备一个纯文本文件，例如：

```text
data/my_corpus.txt
```

然后运行：

```bash
python3 train.py \
  --data_path data/my_corpus.txt \
  --architecture llama \
  --max_steps 1000 \
  --block_size 128 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --out_dir out/my-corpus
```

注意：语料要明显长于 `block_size`，否则无法切出训练和验证 batch。数据越少，模型越容易背诵；数据越干净，loss 和生成效果越稳定。

### 8. 查看训练产物

```bash
ls -lh out/demo
```

训练目录里通常会有：

```text
model.pt          # 模型参数和配置
tokenizer.json    # 字符到 token id 的映射
metadata.json     # 最后一次 loss、模型配置等信息
```

### 9. 常用调试命令

检查 Python 文件是否有语法错误：

```bash
python3 -m py_compile train.py generate.py minillm/*.py
```

强制使用 CPU：

```bash
python3 train.py --device cpu --max_steps 10
```

如果你在 Apple Silicon 上想指定 MPS：

```bash
python3 train.py --device mps --dtype float16 --max_steps 300
```

如果你有 NVIDIA GPU：

```bash
python3 train.py --device cuda --dtype auto --max_steps 300
```

### 10. 一条命令快速复现

想快速从训练到生成，可以依次执行：

```bash
python3 -m pip install -r requirements.txt
python3 train.py --architecture llama --max_steps 300 --block_size 64 --batch_size 16 --gradient_accumulation_steps 2 --out_dir out/demo
python3 generate.py --checkpoint out/demo/model.pt --tokenizer out/demo/tokenizer.json --prompt "To build" --max_new_tokens 200
```

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

### 4. GPT-style vs LLaMA-style

文件：`minillm/model.py`

当前代码可以用 `--architecture` 在两种结构之间切换。

**GPT-style**

```bash
python3 train.py --architecture gpt
```

组件组合：

```text
LayerNorm + GELU MLP + learned position embedding
```

这更接近 GPT-2 / nanoGPT 的教学结构，直观、经典、容易理解。

**LLaMA-style**

```bash
python3 train.py --architecture llama
```

组件组合：

```text
RMSNorm + SwiGLU + RoPE
```

这些组件更接近现代开源 LLM 的常见结构：

- `RMSNorm`：不做均值中心化，只按 root mean square 归一化，结构更简单
- `SwiGLU`：带门控的 MLP，比普通 GELU MLP 表达能力更强
- `RoPE`：把位置信息旋转进 query/key，而不是直接加 learned position embedding

### 5. Causal Self-Attention

普通 self-attention 会让每个位置看到所有 token。但语言模型训练时，位置 `i` 不能看到未来的 token，否则它就作弊了。

所以我们使用 causal mask：

```text
位置 0 只能看 0
位置 1 可以看 0, 1
位置 2 可以看 0, 1, 2
...
```

这就是 decoder-only GPT 能进行自回归生成的关键。

### 6. 自回归生成

文件：`generate.py`

生成时模型一次只预测一个新 token：

```text
prompt -> 预测 next token -> 拼回输入 -> 再预测下一个 token -> ...
```

`temperature` 控制随机性：

- 低 temperature：更保守，更容易重复
- 高 temperature：更随机，更发散

`top_k` 只允许模型从概率最高的 k 个 token 中采样，可以减少很离谱的输出。

### 7. 训练优化

文件：`train.py`

当前训练脚本已经加入了几个真实 LLM 训练里常见的优化点。

**Gradient Accumulation**

如果显存不够，不能直接把 `batch_size` 调大，就可以连续跑多个 microbatch，只累积梯度，最后再更新一次参数：

```text
microbatch 1 -> backward, 不 step
microbatch 2 -> backward, 不 step
microbatch 3 -> backward, 不 step
microbatch 4 -> backward, optimizer.step()
```

这样可以模拟更大的 batch。代码里用 `--gradient_accumulation_steps` 控制。

**Mixed Precision**

在 GPU 或 Apple Silicon 上，部分计算可以使用 `float16` 或 `bfloat16`：

```bash
python3 train.py --device cuda --dtype auto
python3 train.py --device mps --dtype float16
```

混合精度通常可以减少显存占用、提升吞吐。CPU 路径默认保持 `float32`，方便学习和调试。

**AdamW 参数分组**

训练脚本会把参数分成两类：

- 大矩阵权重：使用 weight decay
- bias 和 norm 参数：不使用 weight decay

这是很多 LLM 训练 recipe 中常见的写法，比对所有参数一刀切更合理。

**Tokens/sec**

训练日志里会显示 `tok/s`，也就是每秒处理多少 token。相比只看 step 时间，tokens/sec 更适合比较不同 batch size、context length 和模型大小下的训练效率。

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
2. 加入 KV cache，让生成更快
3. 加入 grouped-query attention，理解 GQA/MQA
4. 加入 LoRA，做参数高效微调
5. 加入简单 eval benchmark，比如困惑度、QA 准确率
6. 写一个 C 或 C++ 推理器，理解权重导出和部署

这条路线会从 nanoGPT 逐渐走向 LLaMA-style mini model，也会覆盖很多面试里常问的 LLM 工程点。
