
import os
import wandb
import hydra
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from aefp.architecture.autoencoder import AutoencoderKL
from aefp.architecture.contrastive_encoder import ContrastiveEncoder
from aefp.trainers import AETrainer, ContrastiveEncoderTrainer
from aefp.datasets.meg import MEGDataset
from aefp.datasets.contrastive import ContrastiveDataset
from aefp.utils.utils import set_seed, save_model, print_dict
from aefp.utils.loss_utils import KLAELoss, CELoss, CrossEntropyLoss
from aefp.utils.testing_utils import test_ae_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg):

    seed = set_seed(cfg.base.seed)
    if cfg.base.seed is None:
        cfg.base.seed = seed

    run = wandb.init(config=OmegaConf.to_container(cfg), **cfg.wandb)
    

    ################################
    ###         DATASET          ###
    ################################
    print("Loading dataset...")
    dataset = MEGDataset(**cfg.dataset.data)

    # If using contrastive learning, wrap the dataset
    if cfg.dataset.contrastive:
        print("Using contrastive dataset...")
        dataset = ContrastiveDataset(base_dataset=dataset, **cfg.dataset.contrastive_params)

    data_loader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True, pin_memory=True)

        
    ################################
    ###          MODEL           ###
    ################################
    print("Creating model...")
    if cfg.model.encoder_only:
        model = ContrastiveEncoder(**cfg.model.params).to(device)
    else:
        model = AutoencoderKL(**cfg.model.params).to(device)
        with torch.no_grad():
            shape = model.encode(dataset[0].unsqueeze(0).unsqueeze(0).to(device)).sample().detach().cpu().shape
            latent_dim = np.prod(shape)
            print("Initialized model with latent dimensionality of:", latent_dim)
        test_ae_sampling(model, dataset, device)
    optimizer = AdamW(model.parameters(), lr=cfg.model.lr)

    
    ##################################
    ###       LOSS & TRAINER       ###
    ##################################
    print("Initializing loss and trainer...")
    if cfg.model.params.lossconfig.loss_type == "KLAELoss":
        loss = KLAELoss(**cfg.model.params.lossconfig.params)
        trainer = AETrainer(
            model, 
            data_loader,
            optimizer, 
            loss, 
            device,
        )
    elif cfg.model.params.lossconfig.loss_type == "CELoss":
        loss = CrossEntropyLoss(**cfg.model.params.lossconfig.params)
        trainer = ContrastiveEncoderTrainer(
            model,
            data_loader,
            optimizer,
            loss,
            device,
        )

    
    ############################
    ###       TRAINING       ###
    ############################
    print("Training...")
    for i in tqdm(range(cfg.model.num_epochs)):
        
        log_dict = trainer.train(split="train")
        tqdm.write(f"Epoch {i+1}/{cfg.model.num_epochs} | Global step {i * len(data_loader)}")

        # log
        print_dict(log_dict, tqdm_write=True)
        wandb.log(log_dict)

        # Save model
        if (i+1) % cfg.base.save_interval == 0:
            save_model(model, cfg, dataset, i+1, ckpt=True, overwrite=True)
        
        # Test model
        if (i+1) % cfg.base.test_interval == 0:
            if not cfg.model.encoder_only:
                test_ae_sampling(model, dataset, device)

            dataset.set_val()
            log_dict = trainer.test()
            dataset.set_train()

            tqdm.write("Validation loss: ", end="")
            print_dict(log_dict, tqdm_write=True)
            wandb.log(log_dict)
        
    run.finish()

    # Save model
    save_model(model, cfg, dataset, cfg.base.save_filename, ckpt=False)


@hydra.main(version_base=None, config_path="../conf/encoder", config_name="omega.yaml")
def main(cfg):
    
    use_fixed_sub_dict = False
    sub_dict_path = "conf/camcan_sub_dict.pt"

    if use_fixed_sub_dict:
        sub_dict = torch.load(sub_dict_path, weights_only=False)
        sub_array = sub_dict["train_sub_ids"] + sub_dict["val_sub_ids"]
        cfg.dataset.data.sub_ids = sub_array
        cfg.dataset.data.val_percent = len(sub_dict["val_sub_ids"]) / len(sub_array)
        print(f"Using fixed subject list with {len(sub_array)} subjects ({len(sub_dict['train_sub_ids'])} train, {len(sub_dict['val_sub_ids'])} val)")

    train(cfg)


if __name__ == "__main__":
    main()
