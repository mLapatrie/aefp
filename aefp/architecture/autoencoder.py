
import torch
import torch.nn as nn

from aefp.architecture.modules import Encoder, Decoder
from aefp.utils.distributions import DiagonalGaussianDistribution

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 kl_beta=0.001,
                 **kwargs
                 ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig) #s = nn.ModuleList([Decoder(**ddconfig) for _ in range(num_decoders)])

        assert ddconfig["double_z"] == True, "KL Divergence is only implemented for double z"

        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.kl_beta = kl_beta

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior