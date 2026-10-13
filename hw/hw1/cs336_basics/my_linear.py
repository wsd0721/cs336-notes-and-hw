from torch import nn
import torch

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 参数初始化
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        sigma = torch.sqrt(torch.tensor(2 / (in_features + out_features)))
        nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T