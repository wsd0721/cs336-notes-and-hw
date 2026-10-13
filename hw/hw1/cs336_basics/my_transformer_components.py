from torch import nn
import torch
from einops import repeat, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 参数初始化
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        sigma = torch.sqrt(torch.tensor(2 / (in_features + out_features)))
        nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 初始化参数
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # 初始化缩放参数
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast，防止溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.scale
        return result.to(in_dtype)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        weights = x1 * torch.sigmoid(x1)
        x2 = x3 * weights
        return self.w2(x2)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 预计算角度
        k_indices = torch.arange(1, d_k / 2 + 1, device=device, dtype=torch.float32)
        freqs = self.theta ** ((-2.0 * k_indices + 2) / self.d_k)
        all_pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(freqs, all_pos, 'half_d, max_seq_len -> max_seq_len half_d')
        # 预计算正余弦值
        cos_half = torch.cos(angles)
        sin_half = torch.sin(angles)
        # 注册buffer，persistent=False，不存进checkpoint
        self.register_buffer("cos_half", cos_half, persistent=False)
        self.register_buffer("sin_half", sin_half, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 取对应的正余弦值序列，shape [seq_len, d_k]
        cos_val = self.cos_half[token_positions]
        sin_val = self.sin_half[token_positions]
        # 计算结果
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x[..., 0::2] * cos_val - x[..., 1::2] * sin_val
        x_rot[..., 1::2] = x[..., 0::2] * sin_val + x[..., 1::2] * cos_val
        return x_rot

def softmax(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    """
    SoftMax函数

    Args:
        x: 输入
        dim: 执行SoftMax的维度

    Returns:
        SoftMax的结果
    """
    # 把待执行的维度放在最后一维，内存连续性小幅提速
    x.transpose(dim, -1)
    max_val = torch.max(x, dim=-1, keepdim=True)[0]
    x = x - max_val
    exp_x = torch.exp(x)
    exp_sum = torch.sum(exp_x, dim=-1, keepdim=True)
    soft_x = exp_x / exp_sum
    return soft_x.transpose(dim, -1)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    缩放点积注意力，支持掩码

    Args:
        Q, K, V：输入向量Query， Key，Value
        mask：掩码，1代表参与score计算

    Returns:
        注意力结果，形状与V相同
    """
    d_k = Q.size(-1)
    scores = einsum(Q, K, 'batch_size ... seq_q d_k, batch_size ... seq_k d_k -> batch_size ... seq_q seq_k')
    logits = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    logits = torch.masked_fill(logits, ~mask, -1e9)
    attn_weights = softmax(logits, -1)
    attn_out = einsum(attn_weights, V, 'batch_size ... seq_q seq_k, batch_size ... seq_k d_v -> batch_size ... seq_q d_v')
    return attn_out