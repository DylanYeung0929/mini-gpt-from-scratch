# 从零构建小型 GPT（作业三）

自建 GPT 架构 + 自训 BPE 分词器 + 预训练 + LoRA 指令微调 + Web 推理演示。

## 目录结构

```
repo/
├── gptfs/                  # 核心包
│   ├── config.py           # 模型/训练配置
│   ├── model.py            # GPT（RMSNorm/RoPE/SwiGLU/GQA/权重共享）+ KV-Cache 生成
│   ├── tokenizer.py        # 字节级 BPE 分词器训练与封装
│   ├── data.py             # 语料 -> uint16 token bin、随机窗口数据集
│   ├── train.py            # 预训练循环（AMP/梯度累积/余弦退火/检查点）
│   ├── lora.py             # 从零实现的 LoRA
│   ├── sft.py              # 指令微调（loss 掩码 + LoRA/全参）
│   ├── inference.py        # 检查点加载 + 流式生成
│   ├── generate.py         # 采样（温度/top-k/top-p/重复惩罚）
│   └── utils.py            # 学习率调度、日志等
├── scripts/
│   ├── prepare_data.py     # 训练分词器 / 生成 token bin
│   ├── download_data.py    # 断点续传下载（支持 hf-mirror）
│   ├── convert_alpaca.py   # Alpaca JSON -> jsonl
│   ├── evaluate.py         # 验证集困惑度 + 样例生成
│   └── app.py              # Gradio 聊天界面
└── artifacts/tokenizer/    # 训练好的分词器
```

## 快速开始

```bash
# 0) 环境（X 盘）
X:/llm-course/venv/Scripts/python.exe -m pip install -r requirements.txt

# 1) 训练分词器（基于语料采样 3 亿字符）
python -m scripts.prepare_data tokenizer --input X:/llm-course/data/raw/train.txt \
    --out artifacts/tokenizer --vocab_size 32768

# 2) 生成 token bin
python -m scripts.prepare_data bins \
    --train_text X:/llm-course/data/raw/TinyStoriesV2-GPT4-train.txt \
    --val_text X:/llm-course/data/raw/TinyStoriesV2-GPT4-valid.txt \
    --out_dir X:/llm-course/data/tokenized

# 3) 预训练
python -m gptfs.train --max_steps 20000 --batch_size 8 --grad_accum 8

# 4) 指令微调（LoRA）
python -m gptfs.sft --ckpt X:/llm-course/checkpoints/pretrain/best.pt \
    --train_jsonl X:/llm-course/data/raw/sft/train.jsonl

# 5) 评估 / Web 演示
python -m scripts.evaluate --ckpt X:/llm-course/checkpoints/sft/sft_merged.pt --sft
python -m scripts.app --ckpt X:/llm-course/checkpoints/sft/sft_merged.pt
```

## 模型默认配置

| 项 | 值 |
|---|---|
| 层数 | 12 |
| d_model | 768 |
| 注意力头 | 12（head_dim 64，MHA） |
| FFN | SwiGLU，hidden=2048 |
| 上下文 | 512 |
| 词表 | 32768（自训 BPE） |
| 位置编码 | RoPE |
| 归一化 | RMSNorm（pre-norm） |
| 权重共享 | 是（token embedding ↔ lm_head） |

参考：Sebastian Raschka, *Build a Large Language Model (From Scratch)*
(https://github.com/rasbt/LLMs-from-scratch)。
