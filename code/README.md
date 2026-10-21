# 手撕代码库

面试高频手撕题的可运行实现。**每个文件都自带自测**，装好依赖后直接跑：

```bash
pip install torch numpy          # 只需要这两个
python code/attention/self_attention.py   # 单独跑某个文件
python code/run_all_tests.py               # 跑全部
```

## 为什么要有这个目录

大模型面试里，八股背得再熟，一让手写就卡住的大有人在。这里的每个实现都按「面试现场能默写出来」的标准写 —— 代码尽量短、不炫技、每一行都能讲清楚为什么。

每个文件的结构都是：

1. **模块顶部注释** —— 面试考点、易错点、常见追问
2. **实现** —— 最小可运行版本
3. **自测** —— 用断言验证正确性，跑一遍就知道对不对

## 目录

| 文件 | 内容 | 面试频率 |
|---|---|---|
| [attention/self_attention.py](attention/self_attention.py) | 缩放点积注意力 + causal mask | ★★★★★ |
| [attention/mha.py](attention/mha.py) | 多头注意力（含拆分/合并、参数量） | ★★★★★ |
| [attention/gqa.py](attention/gqa.py) | GQA / MQA，KV 头复用 | ★★★★☆ |
| [attention/flash_attention_naive.py](attention/flash_attention_naive.py) | 分块 + 在线 softmax | ★★★★☆ |
| [position/rope.py](position/rope.py) | 旋转位置编码，验证相对位置性质 | ★★★★★ |
| [inference/kv_cache.py](inference/kv_cache.py) | KV Cache 增量解码 + 显存计算 | ★★★★★ |
| [inference/sampling.py](inference/sampling.py) | temperature / top-k / top-p | ★★★★☆ |
| [peft/lora.py](peft/lora.py) | LoRA 层、初始化、merge | ★★★★★ |
| [norm/rmsnorm.py](norm/rmsnorm.py) | RMSNorm / LayerNorm | ★★★★☆ |
| [loss/dpo.py](loss/dpo.py) | DPO loss 与梯度直觉 | ★★★☆☆ |
| [loss/grpo.py](loss/grpo.py) | GRPO 组内归一化优势 | ★★★★☆ |
| [tokenizer/bpe.py](tokenizer/bpe.py) | BPE 合并训练 | ★★★☆☆ |

## 使用建议

- **先自己写，再对照** —— 直接看答案会产生「我会了」的错觉，实际默写时还是会卡。
- **注释当成口述稿** —— 每个函数的 docstring 就是被问到时该怎么讲。
- **改数字跑一遍** —— 比如把 `d_k` 从 64 改成 1024，看数值是否真的爆炸，理解「为什么要除 √d_k」。
