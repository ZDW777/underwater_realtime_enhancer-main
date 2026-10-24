"""复制到 models/ 并保存为 ExampleNet.py，文件名必须与主网络类名一致。"""
from __future__ import annotations

from torch import nn


REQUIRE_WEIGHTS = True


class ExampleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 替换为训练权重所对应的完整网络结构。
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)
