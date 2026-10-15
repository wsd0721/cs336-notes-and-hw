import torch
from torch import einsum
from einops import reduce

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    实现交叉熵损失函数，用于LLM训练
    Args:
        logits: 原始logits分数，形状[batch_size, seq_len, vocab_size]
        targets: GT，形状[batch_size, seq_len]
    Returns:
        交叉熵计算结果，ℓ𝑖 = − log softmax(𝑜𝑖)[𝑥𝑖+1]
    """
    # 数值稳定性：减去最大值
    max_value = logits.max(dim=-1, keepdim=True)[0]
    logits = logits - max_value
    # 计算ℓ𝑖，分子log和e抵消
    exp_logits = torch.exp(logits)
    exp_sum = reduce(exp_logits, 'l (one v) -> l one', 'sum', one=1)
    log_exp_sum = torch.log(exp_sum)
    targets = targets.unsqueeze(dim=-1)
    target_value = logits.gather(-1, targets)
    l = -target_value + log_exp_sum
    # 求均值
    l = l.squeeze(dim=-1)
    ce = l.mean(dim=-1, keepdim=False)
    return ce