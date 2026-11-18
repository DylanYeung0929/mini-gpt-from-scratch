"""Gradio web demo for the fine-tuned model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch

from gptfs.generate import build_chat_prompt
from gptfs.inference import load_model, stream_generate
from gptfs.utils import get_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="X:/llm-course/checkpoints/sft_bi/sft_merged.pt")
    p.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer_bi")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)

    def respond(message, history, max_new_tokens, temperature, top_p):
        if not message.strip():
            yield ""
            return
        prompt = build_chat_prompt(message)  # 与 SFT 训练模板保持一致（无 system 段）
        for text in stream_generate(model, tok, prompt,
                                    max_new_tokens=int(max_new_tokens),
                                    temperature=temperature, top_p=top_p,
                                    repetition_penalty=1.1, device=device):
            yield text

    with gr.Blocks(title="GPT-from-scratch 演示") as demo:
        gr.Markdown("## 从零构建的小型 GPT 指令模型\n"
                    "基于自训 BPE 分词器 + 自建 GPT 架构，TinyStories 预训练 + LoRA 指令微调。")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(label="输入", placeholder="输入你的问题，回车发送…",
                                 lines=2)
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空")
            with gr.Column(scale=1):
                max_tokens = gr.Slider(16, 512, value=200, step=16, label="最大生成长度")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="温度")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")

        def chat(message, history, max_tokens, temperature, top_p):
            history = history or []
            history = history + [{"role": "user", "content": message}]
            partial = ""
            for partial in respond(message, history, max_tokens, temperature, top_p):
                yield history + [{"role": "assistant", "content": partial}], ""
            yield history + [{"role": "assistant", "content": partial}], ""

        send.click(chat, [msg, chatbot, max_tokens, temperature, top_p],
                   [chatbot, msg])
        msg.submit(chat, [msg, chatbot, max_tokens, temperature, top_p],
                   [chatbot, msg])
        clear.click(lambda: ([], ""), None, [chatbot, msg])

    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
