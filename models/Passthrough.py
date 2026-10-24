from __future__ import annotations

import torch
from torch import nn


REQUIRE_WEIGHTS = False


class Passthrough(nn.Module):
    """无权重的流程测试模型，输出与输入完全相同。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
