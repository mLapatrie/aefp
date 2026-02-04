import torch

from tqdm import tqdm

from aefp.utils.loss_utils import CELoss, CrossEntropyLoss

class ContrastiveEncoderTrainer:

    def __init__(
            self,
            encoder,
            data_loader,
            optimizer,
            loss_fn,
            device,
    ):
        self.encoder = encoder
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, backprop=True, split="train"):
        self.encoder.train()

        for batch in tqdm(self.data_loader, leave=False):
            batch1 = batch[0].unsqueeze(1).to(self.device)
            batch2 = batch[1].unsqueeze(1).to(self.device)
            ys = batch[2].to(self.device)
            sub_idx = batch[3].to(self.device)

            self.optimizer.zero_grad()

            z1 = self.encoder(batch1).reshape(batch1.shape[0], -1)
            z2 = self.encoder(batch2).reshape(batch2.shape[0], -1)

            if isinstance(self.loss_fn, CELoss):
                loss, log_dict = self.loss_fn(z1, z2, ys)
            elif isinstance(self.loss_fn, CrossEntropyLoss):
                loss, log_dict = self.loss_fn(z1, sub_idx)
            else:
                raise ValueError("Loss function not recognized for ContrastiveEncoderTrainer.")

            if backprop:
                loss.backward()
                self.optimizer.step()

            log_dict[f"{split}_total_loss"] = loss

        return log_dict
    
    def test(self):
        self.encoder.eval()
        with torch.no_grad():
            log_dict = self.train(backprop=False, split="val")
        self.encoder.train()

        return log_dict


class AETrainer:

    def __init__(
            self,
            ae,
            data_loader,
            optimizer,
            loss_fn,
            device,
    ):
        self.ae = ae
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, backprop=True, split="train"):

        self.ae.train()

        for batch in tqdm(self.data_loader, leave=False):
            
            batch = batch.unsqueeze(1).to(self.device)

            x_hat, posteriors = self.ae(batch)
    
            # train ae
            ae_loss, log_dict = self.loss_fn(batch, x_hat, posteriors, split=split) 
            
            if backprop:
                self.optimizer.zero_grad()
                ae_loss.backward()
                self.optimizer.step()

            log_dict[f"{split}_total_loss"] = ae_loss
            
        return log_dict
    
    def test(self):
        self.ae.eval()
        with torch.no_grad():
            log_dict = self.train(backprop=False, split="val")
        self.ae.train()

        return log_dict