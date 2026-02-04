
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
from aefp.utils.utils import set_seed, save_model, print_dict, load_autoencoder
from aefp.utils.loss_utils import KLAELoss, CELoss
from aefp.utils.testing_utils import test_ae_sampling
from aefp.utils.fingerprinting_utils import get_valid_test_subjects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune(model, cfg, sub_ids, epochs=10):

    run = wandb.init(config=OmegaConf.to_container(cfg), **cfg.wandb)
    
    ################################
    ###         DATASET          ###
    ################################
    print("Loading dataset...")

    cfg.dataset.data.sub_ids = sub_ids
    cfg.dataset.data.val_percent = 0.0  # No validation set for fine-tuning

    dataset = MEGDataset(**cfg.dataset.data)
    data_loader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True, pin_memory=True)

    ################################
    ###          MODEL           ###
    ################################
    print("Using provided model...")
    optimizer = AdamW(model.parameters(), lr=cfg.model.lr)

    
    ##################################
    ###       LOSS & TRAINER       ###
    ##################################
    print("Initializing loss and trainer...")
    loss = KLAELoss(**cfg.model.params.lossconfig.params)
    loss.current_step = np.inf # Skip warmup schedules
    trainer = AETrainer(
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
    for i in tqdm(range(epochs), desc="Epochs"):
        
        log_dict = trainer.train(split="train")
        tqdm.write(f"Epoch {i+1}/{cfg.model.num_epochs} | Global step {i * len(data_loader)}")

        # log
        print_dict(log_dict, tqdm_write=True)
        wandb.log(log_dict)
        
    run.finish()

    # Save model
    save_filename = cfg.base.save_filename.split(".")[0] + f"_ft.pt"
    save_model(model, cfg, dataset, save_filename, ckpt=False)


if __name__ == "__main__":
    dataset = "camcan"
    autoencoder_path = "/export01/data/camcan/saved_models/aefp/autoencoder/autoencoder_200.pt"
    
    ae, cfg, sub_dict = load_autoencoder(autoencoder_path)
    ae.to(device)

    test_sub_ids, _ = get_valid_test_subjects(
        sub_dict=sub_dict,
        root_dir=cfg.dataset.data.data_path,
        same_session=True,
    )

    epochs = 10

    fine_tune(ae, cfg, test_sub_ids, epochs=epochs)
