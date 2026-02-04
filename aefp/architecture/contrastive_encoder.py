
import torch
import torch.nn as nn

from aefp.architecture.modules import Encoder


class ContrastiveEncoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 embed_shape=40000,
                 class_head=False,
                 num_classes=190,
                 **kwargs
                 ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)

        self.profiling = False

        double = 2 if ddconfig["double_z"] else 1

        self.quant_conv = nn.Conv2d(double*ddconfig["z_channels"], double*embed_dim, 1)
        self.embed_dim = embed_dim

        self.class_head = class_head
        self.num_classes = num_classes

        self.contrastive_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_shape, num_classes) if class_head else nn.Identity(),
        )

    def forward(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)

        if self.profiling:
            z = torch.flatten(moments, start_dim=1)
            return z

        z = self.contrastive_head(moments)
        return z

    def set_profiling(self, profiling: bool):
        self.profiling = profiling