from torch import nn
import torch
from einops import repeat, einsum, rearrange

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

class MutliHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, apply_rope: bool = True, theta: int = 10000, max_seq_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.apply_rope = apply_rope
        # 线性投影矩阵
        self.qkv = Linear(d_model, 3 * d_model)
        self.proj = Linear(d_model, d_model)
        if self.apply_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # QKV投影 + 多头拆分
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'batch_size seq_len (three h d_k) -> three batch_size h seq_len d_k', three = 3, h = self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        seq_len = q.size(-2)
        # 应用RoPE
        if self.apply_rope:
            token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        # 带因果掩码的自注意力
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        v_out = scaled_dot_product_attention(q, k, v, mask)
        # 合并投影回d_model维
        v_out = rearrange(v_out, 'batch_size h seq_len d_k -> batch_size seq_len (h d_k)')
        out = self.proj(v_out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int, max_seq_len: int):
        super().__init__()
        # 三大组件：注意力，FFN，norm
        self.attn = MutliHeadSelfAttention(d_model, num_heads, True, theta, max_seq_len)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention部分：norm，attn，res
        x_tmp = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + x_tmp
        # FFN部分：norm，ffn，res
        x_tmp = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + x_tmp

        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()
        # 组件：embedding，TransformerBlock，norm，LM Head
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x