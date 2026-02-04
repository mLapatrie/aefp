import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from aefp.architecture.autoencoder import AutoencoderKL
from aefp.architecture.contrastive_encoder import ContrastiveEncoder
from aefp.utils.utils import load_autoencoder
from aefp.utils.fingerprinting_utils import get_valid_test_subjects
from aefp.utils.fingerprinting_utils import upper_aec_torch, flat_psd_torch


def _latent_from_window(batch, model, method, device):
    x = batch.unsqueeze(1).to(device)  # (1,1,C,window)
    match method:
        case "aec":
            out = upper_aec_torch(x.squeeze(1))
        case "psd":
            out = flat_psd_torch(x.squeeze(1))
        case "ae":
            with torch.no_grad():
                out = model.encode(x).sample()
        case "ae_ft":
            with torch.no_grad():
                out = model.encode(x).sample()
        case "cont":
            with torch.no_grad():
                out = model(x)
        case _:
            raise ValueError(f"Unknown method: {method}")
        
    return out.cpu().detach().numpy().reshape(-1)


def extract_subject_embeddings(
    root_dir: str,
    model_path: str,
    method: str,
    out_dir: str,
    same_session: bool = True,
    data_modality: str = "meg",
    data_type: str = "rest",
    data_space: str = "source_200",
    block_channels: int = 200,
    window_size: int = 900,
    segment_size: int = 4500,   # kept for API compatibility; unused now
    step_size: int = 150,
    train: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Extract per-window embeddings for all subjects using sliding windows.

    Parameters
    ----------
    root_dir : str
        Path to dataset root.
    model_path : str
        Path to a trained autoencoder or encoder.
    out_dir : str
        Directory where embeddings will be stored.
    data_modality, data_type, data_space : str
        Dataset naming parameters.
    block_channels : int
        Number of channels per block fed to the network.
    window_size : int
        Window length in samples (default 900 -> 6 s at 150 Hz).
    step_size : int
        Step between window starts in samples (default 150 -> 1 s at 150 Hz).
        If step_size < window_size, windows overlap.
    device : torch.device
        Device used for encoding.
    """

    model, cfg, sub_dict = load_autoencoder(model_path)
    model.to(device)
    model.eval()

    if train:
        out_dir = os.path.join(out_dir, "train")
    os.makedirs(out_dir, exist_ok=True)

    test_sub_ids, _ = get_valid_test_subjects(sub_dict, root_dir, same_session=same_session)
    test_sub_ids = sub_dict["train_sub_ids"] if train else test_sub_ids
    print("USING TRAIN DATA" if train else "USING TEST DATA")
    test_sub_ids = test_sub_ids

    for sub in tqdm(test_sub_ids, desc="Subjects"):
        session_paths = sorted(
            glob(os.path.join(root_dir, sub, "ses-*", data_modality, data_type, f"{data_space}.pt"))
        )

        all_latents = []

        for ses_path in session_paths:
            data = torch.load(ses_path, weights_only=False)  # (C, T)
            data = data[:block_channels]
            C, T = data.shape

            # Sliding window starts using step_size
            win_starts = range(0, max(0, T - window_size + 1), step_size)
            if not win_starts:
                continue

            if method in {"psd", "aec"}:
                block = data.unsqueeze(0)
                latent = _latent_from_window(block, model, method, device)
                all_latents.append(latent)
                continue

            latents = []
            with torch.no_grad():
                for s in win_starts:
                    e = s + window_size
                    block = data[:, s:e]

                    # z-score normalize per window (robust to tiny std)
                    block = (block - block.mean()) / (block.std() + 1e-8)

                    x = block.unsqueeze(0) # (1,1,C,window)
                    z = _latent_from_window(x, model, method, device)

                    latents.append(z)

            if latents:
                all_latents.append(np.stack(latents, axis=0))

        if all_latents:
            subj_arr = np.concatenate(all_latents, axis=0)  # (N_windows_total, D)
            np.save(os.path.join(out_dir, f"{sub}.npy"), subj_arr)


if __name__ == "__main__":

    dataset = "camcan"
    same_kernel = True
    root_dir = f"/export01/data/{dataset}/"

    train = False

    method = "aec"
    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"
    if method == "cont":
        model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200.pt"
        
    out_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/{dataset}{"_sk" if same_kernel else ""}/{method}")
    data_modality = "meg"
    data_type = "rest"
    data_space = f"source_200{'_sk' if same_kernel else ''}"
    window_size = 900
    segment_size = 900
    step_size = 150 # 1s

    extract_subject_embeddings(
        root_dir=root_dir,
        model_path=model_path,
        method=method,
        out_dir=out_dir,
        data_modality=data_modality,
        data_type=data_type,
        data_space=data_space,
        window_size=window_size,
        segment_size=segment_size,
        step_size=step_size,
        train=train,
    )
