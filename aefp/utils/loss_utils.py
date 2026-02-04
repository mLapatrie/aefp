
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KLAELoss(nn.Module):
    
    def __init__(self, kl_weight_start=1e-6, kl_weight_end=1e-4, kl_warmup_steps=10000, kl_warmup_start=1000, **kwargs):

        super().__init__()

        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_warmup_start = kl_warmup_start
        self.current_step = 0

        self.kl_weight_schedule = torch.linspace(kl_weight_start, kl_weight_end, kl_warmup_steps)

    def forward(self, x, x_hat, posteriors, split="train", **kwargs):

        ## 1. compute the MSE loss between the input and the output
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')

        # 2. compute the KL divergence between the approximate posterior and the prior
        kl_loss = torch.mean(posteriors.kl())
    
        kl_weight = self.kl_weight_start
        if self.current_step > self.kl_warmup_start:
            kl_weight = self.kl_weight_schedule[self.current_step - self.kl_warmup_start] \
                if self.current_step - self.kl_warmup_start < self.kl_warmup_steps else self.kl_weight_end

        total_loss = mse_loss + kl_weight * kl_loss
        self.current_step += 1

        log_dict = {
            f"{split}_total_loss": total_loss, 
            f"{split}_mse_loss": mse_loss, 
            f"{split}_kl_loss": kl_loss,
            f"{split}_kl_per_dim": kl_loss / np.prod(posteriors.mean.shape[1:]), 
        }

        return total_loss, log_dict
    

class CELoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super().__init__()
        self.margin = margin
        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, z1, z2, ys, split="train", **kwargs):
        loss = self.loss_fn(z1, z2, ys)
        log_dict = {f"{split}_total_loss": loss}
        return loss, log_dict


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, split="train", **kwargs):
        loss = self.loss_fn(logits, labels)
        log_dict = {f"{split}_total_loss": loss}
        return loss, log_dict