# miniLLM

这是一个从 0 开始实现的极简 GPT 语言模型。它的目标不是训练出很强的大模型，而是把 LLM 最核心的一条链路完整跑通：

```text
文本语料 -> tokenizer -> token ids -> batch -> Transformer -> loss -> 反向传播 -> checkpoint -> 自回归生成
```

当前版本参考 nanoGPT 的思路，实现了一个 **decoder-only Transformer**。模型支持两种架构配置：

- `--architecture gpt`：GPT-2 风格，LayerNorm + GELU MLP + learned position embedding
- `--architecture llama`：LLaMA 风格，RMSNorm + SwiGLU + RoPE

tokenizer 支持两种实现：

- `--tokenizer char`：字符级 tokenizer，每个字符一个 token，最透明
- `--tokenizer bpe`：教学版 BPE tokenizer，从字符开始合并高频相邻片段，序列更短

## 文件结构

```text
miniLLM/
├── data/sample.txt          # 一个很小的示例训练语料
├── minillm/
│   ├── data.py              # 读取文本、切分 train/val、构造 next-token batch
│   ├── model.py             # GPTConfig、CausalSelfAttention、Block、MiniGPT
│   └── tokenizer.py         # CharTokenizer 和 BPETokenizer
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
python train.py --tokenizer char --architecture gpt --max_steps 300 --block_size 64 --batch_size 16
```

用 BPE tokenizer 训练一个玩具模型：

```bash
python train.py --tokenizer bpe --bpe_vocab_size 200 --architecture gpt --max_steps 300 --block_size 32 --batch_size 16
```

训练一个 LLaMA-style 玩具模型：

```bash
python train.py --architecture llama --max_steps 300 --block_size 64 --batch_size 16
```

生成文本：

```bash
python generate.py --prompt "To build" --max_new_tokens 200
```

关闭 KV cache，对比慢速生成路径：

```bash
python generate.py --prompt "To build" --max_new_tokens 200 --disable_kv_cache
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
  --tokenizer char \
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
- `--tokenizer`：tokenizer 类型，`char` 或 `bpe`
- `--bpe_vocab_size`：BPE 目标词表大小，只在 `--tokenizer bpe` 时使用
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
- `--disable_kv_cache`：关闭 KV cache，回到“每步重算整段上下文”的教学路径

### 7. 换成你自己的训练语料

准备一个纯文本文件，例如：

```text
data/my_corpus.txt
```

然后运行：

```bash
python3 train.py \
  --data_path data/my_corpus.txt \
  --tokenizer bpe \
  --bpe_vocab_size 1000 \
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
python3 train.py --tokenizer bpe --bpe_vocab_size 200 --architecture llama --max_steps 300 --block_size 32 --batch_size 16 --gradient_accumulation_steps 2 --out_dir out/demo
python3 generate.py --checkpoint out/demo/model.pt --tokenizer out/demo/tokenizer.json --prompt "To build" --max_new_tokens 200
```

## 核心概念

### 1. Tokenizer

文件：`minillm/tokenizer.py`

模型不能直接处理字符串，所以 tokenizer 负责：

```text
"hello" -> [token_id_1, token_id_2, ...]
```

当前实现保留了两个 tokenizer，方便对比。

**CharTokenizer**

- 统计训练语料中出现过的所有字符
- 给每个字符分配一个整数 id
- 训练时把文本变成 token id 序列
- 生成后把 token id 还原成字符串

示例：

```text
"hi!"  -> ["h", "i", "!"] -> [2, 3, 1]
"你好" -> ["你", "好"]     -> [4, 5]
```

**BPETokenizer**

BPE 会从字符级 token 开始，不断合并语料里最常见的相邻 token pair。

示例：

```text
训练文本: "low lower lowest"

初始:
["l", "o", "w", " ", "l", "o", "w", "e", "r", ...]

合并 ("l", "o") -> "lo":
["lo", "w", " ", "lo", "w", "e", "r", ...]

继续合并 ("lo", "w") -> "low":
["low", " ", "low", "e", "r", ...]
```

所以同样是 `"lower"`：

```text
CharTokenizer: "lower" -> ["l", "o", "w", "e", "r"]
BPETokenizer:  "lower" -> ["low", "e", "r"] 或 ["low", "er"]
```

BPE token 数更少，因此训练和推理通常更省。char-level 的优点是实现透明，BPE 的优点是更接近真实 LLM。

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

### 7. KV Cache

文件：`minillm/model.py`

这一版实现已经加入了教学版 KV cache。

**为什么需要 KV cache**

没有 KV cache 时，生成流程是：

```text
prompt
-> 算整段上下文的 attention
-> 生成 1 个新 token
-> 把新 token 拼回去
-> 再把整段上下文重算一遍
```

这样做最大的问题是：旧 token 的 key/value 每一步都在重复计算。

有 KV cache 时：

```text
prefill: 先把 prompt 完整跑一遍，缓存每层的 K/V
decode: 之后每次只输入 1 个新 token，直接复用历史 K/V
```

所以 KV cache 的核心直觉是：

```text
旧 token 的 K/V 不变，就不要重复算
```

**这一版代码里的实现层次**

- `CausalSelfAttention`：支持传入 `past_kv`，并返回新的 `present_kv`
- `Block`：把每层 attention 的 cache 往上层传
- `MiniGPT.forward_with_kv_cache()`：推理专用前向，返回 logits 和整模型的 cache
- `MiniGPT.generate()`：默认开启 KV cache，也保留了关闭 cache 的慢速路径

**Prefill 和 Decode**

- `prefill`：把整段 prompt 输入模型，建立所有层的 KV cache
- `decode`：后续每一步只输入最新生成的 1 个 token

这是理解真实 LLM 推理延迟的关键切分。

**为什么还要处理 block_size 上限**

当前模型和旧实现一样，只保留最近的 `block_size` 个 token 作为上下文窗口。  
所以当缓存长度达到窗口上限后，代码会用“最近窗口”重建一次 cache，保证语义和
原来“每次都截取最近窗口重算”的实现一致。

你可以直接对比两条生成命令：

```bash
python3 generate.py --checkpoint out/demo/model.pt --tokenizer out/demo/tokenizer.json --prompt "To build" --max_new_tokens 200
python3 generate.py --checkpoint out/demo/model.pt --tokenizer out/demo/tokenizer.json --prompt "To build" --max_new_tokens 200 --disable_kv_cache
```

### 8. 训练优化

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

## 面试问题

这一步实现完之后，你可以比较自然地回答这些问题：

1. 什么是 KV cache，它缓存了什么？
   - 缓存的是每一层 attention 的 key/value，不是整个 hidden state。

2. 为什么 KV cache 能加速推理？
   - 因为旧 token 的 K/V 在后续 decode 中不会变，不需要重复计算。

3. 为什么训练时一般不用 KV cache？
   - 训练时要并行处理整段序列，重点是一次性算完所有位置；KV cache 更适合自回归生成阶段。

4. prefill 和 decode 有什么区别？
   - prefill 输入整段 prompt，建立缓存；decode 每步只输入 1 个 token。

5. KV cache 的代价是什么？
   - 更省算力，但会占显存/内存，因为要保存每层每个历史 token 的 K/V。

6. 为什么上下文越长，KV cache 占用越大？
   - 因为历史 token 越多，要缓存的 K/V 张量就越长。

7. 如果上下文超过 block size 怎么办？
   - 当前实现按最近窗口重建 cache，保持和原始最近窗口语义一致。

8. KV cache 和 tokenizer 有什么关系？
   - tokenizer 影响序列长度；BPE 往往让 token 更少，因此同样文本下 KV cache 也更省。

## 你可以怎么继续扩展

建议按这个顺序升级：

1. 把教学版 BPE 升级成 byte-level BPE，减少未知字符问题
2. 加入 grouped-query attention，理解 GQA/MQA
3. 加入 LoRA，做参数高效微调
4. 加入简单 eval benchmark，比如困惑度、QA 准确率
5. 写一个 C 或 C++ 推理器，理解权重导出和部署

这条路线会从 nanoGPT 逐渐走向 LLaMA-style mini model，也会覆盖很多面试里常问的 LLM 工程点。
