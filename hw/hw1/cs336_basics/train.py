import torch
from torch import einsum
from einops import reduce
from math import sqrt

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

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: int, weight_decay: int, betas: tuple[float, float], eps: float):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            # 注意一个p是一个nn.Parameter对象，不一定只有一个参数值
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get state and grad
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # compute new state
                t += 1
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad * grad)
                lr_mod = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # weight_decay
                p.data -= lr * wd * p.data
                # gradient descend
                p.data -= lr_mod * m / (torch.sqrt(v) + eps)
                # update state
                state["t"] = t
                state["m"] = m
                state["v"] = v