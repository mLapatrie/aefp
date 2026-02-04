
import os
import torch
import numpy as np

from tqdm import tqdm

from aefp.architecture.autoencoder import AutoencoderKL
from aefp.architecture.contrastive_encoder import ContrastiveEncoder


def set_seed(seed):
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_save_filename(base_path, filename, overwrite=False):
    save_path = os.path.join(base_path, filename)
    copy_counter = 1
    while os.path.exists(save_path) and not overwrite:
        save_path = os.path.join(base_path, f"{filename.split('.')[0]}_{copy_counter}.{filename.split('.')[1]}")
        copy_counter += 1
    return save_path


def save_model(model, cfg, dataset, filename, ckpt=False, overwrite=False, verbose=True):
    if ckpt:
        save_path = get_save_filename(os.path.join(cfg.base.save_dir, "ckpt"), "ckpt_" + str(filename) + ".pt", overwrite=overwrite)
    else:
        save_path = get_save_filename(os.path.join(cfg.base.save_dir), str(filename), overwrite=overwrite)
    
    sub_dict = {"train_sub_ids": dataset.train_sub_ids, "val_sub_ids": dataset.val_sub_ids}
        
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "sub_dict": sub_dict}, save_path)
    if verbose:
        print("Saved model at", save_path)


def print_dict(log_dict, tqdm_write=False):
    text = ""
    for key, value in log_dict.items():
        if tqdm_write:
            text += f"{key}: {value.item()}, "
        else:
            print(f"{key}: {value.item()}", end=", ")
    
    if tqdm_write:
        tqdm.write(text)
        return
    print()
    

def load_autoencoder(autoencoder_path, method="ae"):

    model_dict = torch.load(autoencoder_path, map_location=torch.device("cpu"), weights_only=False)
    sub_dict = model_dict["sub_dict"]
    cfg = model_dict["cfg"]

    if cfg["model"]["encoder_only"]:
        model = ContrastiveEncoder(class_head=method=="cross", **model_dict["cfg"]["model"]["params"])
        model.set_profiling(True)
    else:
        model = AutoencoderKL(**model_dict["cfg"]["model"]["params"])
    model.load_state_dict(model_dict["state_dict"])
    
    return model, cfg, sub_dict