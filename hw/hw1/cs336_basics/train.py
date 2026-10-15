import torch
from torch import einsum
from einops import reduce
from math import sqrt, cos, pi

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

def coine_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    带warmup的余弦学习率调度器
    Args:
        it: 迭代次数
        max_learning_rate: 最大学习率
        min_learning_rate: 最小学习率
        warmup_iters: warmup迭代数
        cosine_cycle_iters: 余弦调度结束的迭代数

    Returns:
        第it次迭代对应的学习率
    """
    lr = 0.0
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr

if __name__ == '__main__':
    batch_size = 1024
    steps = 4e5
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 4288 # the nearest multiple of 64 to 8/3 × 1600
    d_k = d_model // num_heads
    # 内存核算：parameters, activations, gradients, and optimizer states
    # parameters
    num_parameter_embedding = vocab_size * d_model
    # RMSNorm，QKV矩阵，proj矩阵，RMSNorm，FFN矩阵
    num_parameter_transformer = d_model + d_model * (3 * d_model) + d_model * d_model + d_model + 3 * d_model * d_ff
    num_parameter = num_parameter_embedding + num_layers * (num_parameter_transformer) + d_model + num_parameter_embedding
    # gradients
    num_gradient = num_parameter
    # optimizer states
    num_optim_state = 2 * num_parameter
    # activations
    num_activation_embedding = context_length * d_model
    # RMSNorm, QKV矩阵, QKT/sqrt(d_k), SoftMax, v_out, out_proj, w1,w2,w3,swish, gate
    num_activation_transformer = 2 * context_length * d_model + 3 * context_length * d_model + num_heads * context_length * context_length + num_heads * context_length * context_length + context_length * d_model + context_length * d_model + 4 * context_length * d_ff + context_length * d_model
    num_activation_fn_norm = context_length * d_model
    num_activation_lmhead = context_length * vocab_size
    num_activation_ce = context_length
    num_activation = num_activation_embedding + num_layers * num_activation_transformer + num_activation_fn_norm + num_activation_lmhead + num_activation_ce
    memory_usage = 4 * (num_parameter + num_gradient + num_optim_state + num_activation) * batch_size
    print(f'峰值内存占用：{memory_usage}B，合{memory_usage / 1e9}GB')

    # FLOPS核算：仅计算AdamW
    flops_newm = 3 * num_parameter
    flops_newv = 4 * num_parameter
    flops_mod = 2 * num_parameter
    flops_descend = 7 * num_parameter
    # 16 FLOPs for each parameter
    flops_adamw = flops_newm + flops_newv + flops_mod + flops_descend
    # FLOPS为2.6247e+10，相比前向/反向过程，可以忽略
    print(f'一次AdamW的FLOPS为：{flops_adamw}次，合{flops_adamw/1e9}GFLOPS')

    # 训练时间
    # 完整前向FLOPS，包含前向，反向，优化器三部分，其中优化器可以忽略，根据缩放定律，逐元素操作相比矩阵乘法也可以忽略
    # 四个d到d投影8Ld^2，注意力两个大乘法4L^2d + FFN 6Ldd_ff
    flops_transformer = 2 * context_length * d_model * (3 * d_model) + num_heads * 2 * context_length * d_k * context_length + num_heads * 2 * context_length * context_length * d_k + 2 * context_length * d_model * d_model + 3 * 2 * context_length * d_model * d_ff
    flops_lmhead = 2 * context_length * vocab_size * d_model
    flops_forward = num_layers * flops_transformer + flops_lmhead
    flops_step = flops_forward * 3 * batch_size
    efficiency = 0.5 * 495 * 1e12
    time_all = flops_step * steps / efficiency
    # 总训练FLOPS，按照6NP计算为4.1283e+21, 实际计算为4.3214e+21, 误差为4.47%，非常接近
    print(f'估计模型训练一轮的FLOPS为：{6 * num_parameter * batch_size * context_length}次，合{6 * num_parameter * batch_size * context_length/1e9}GFLOPS')
    print(f'模型训练一轮的FLOPS为：{flops_step}次，合{flops_step/1e9}GFLOPS')
    print(f'模型总训练时间为: {time_all/3600}h')  # 4850.06h