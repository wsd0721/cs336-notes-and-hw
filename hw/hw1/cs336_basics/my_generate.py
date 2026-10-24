import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import torch
import argparse
from my_transformer_components import softmax, TransformerLM
from my_bpe_tokenizer import Tokenizer
from einops import pack

def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eot_id: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    返回完整序列：prompt + 生成续写部分，shape [B, T + new_len]

    Args:
        model：模型
        eot_id：<|endoftext|> 的token id
        prompt_ids：提示词序列，dtype=long
        max_new_tokens：最大token数
        temperature：温度系数
        top_p：topP阈值
    
    Returns：
        生成结果token序列
    """
    # 注意开推理模式和关闭梯度
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            # 推理，取最后一行
            logits = model(prompt_ids)[:, -1, :]
            # 温度系数
            logits /= temperature
            # softmax
            probs = softmax(logits, dim=-1)
            # Top-P过程
            sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
            sum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = sum_probs >= top_p
            # 注意不能原地赋值
            mask = torch.cat([
                torch.zeros_like(mask[..., :1]),
                mask[..., :-1]
            ], dim=-1)
            sorted_probs[mask] = 0.0
            # 注意要归一化 + 转换回原始顺序
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sorted_probs = torch.scatter(sorted_probs, -1, indices, sorted_probs)
            # 采样
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            # 拼接
            prompt_ids = torch.cat([prompt_ids, next_token], dim=-1)
            # 识别结束token
            if next_token == eot_id:
                break
    return prompt_ids

def main():
    parser = argparse.ArgumentParser()
    # 模型超参
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    prompt = args.prompt
    model_path = Path(args.model_path)
    device = args.device
    tokenizer = Tokenizer.from_files('/home/wsd/diy-llm/cs336-notes-and-hw/hw/hw1/outputs/vocab.json', '/home/wsd/diy-llm/cs336-notes-and-hw/hw/hw1/outputs/merges.json')
    prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, :].to(device)
    model = TransformerLM(1600, 25, 4288, 10000, 999, 1024, 48).to(device)
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {model_path}")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model_state_dict'])
    output = generate(model, prompt, 3, 256)
    print(tokenizer.decode(output[0].cpu().tolist()))

if __name__ == '__main__':
    main()