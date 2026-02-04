import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from fingerprint import main as fingerprint_main
from helpers.fingerprinting_helpers import bootstrap_metrics
from aefp.utils.plotting_utils import LABELS


def main():

    use_fake_data = False

    skip_compute = True
    skip_fp = True

    methods = ["ae", "cont", "cross"]

    dataset = "camcan"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}"

    if use_fake_data:
        base_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.tmp/fake/windows/")
    else:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fingerprint_lengths/{dataset}/")
    fig_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.figures/")
    latent_base_dir = os.path.join(base_save_dir, "latent_vectors")
    results_base_dir = os.path.join(base_save_dir, "results")
    os.makedirs(latent_base_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    data_modality = "meg"
    data_type = "rest"
    data_space = "source_200_sk"
    same_session = True
    band = None
    bootstraps = 1
    bootstraps_size = 1
    fs = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window_size_sec = 6
    window_size = window_size_sec * fs
    segment_size_sec = 30
    segment_size = segment_size_sec * fs
    step_size = 1  # Precompute every possible window

    results = {}
    for method in methods:
        method_key = method
        if method == "cont":
            model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200.pt"
        elif method == "cross":
            model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200_ce.pt"
        else:
            model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"
        run_method = method

        latent_dir = os.path.join(latent_base_dir, method_key)
        results_dir = os.path.join(results_base_dir, method_key)
        os.makedirs(latent_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Compute latent vectors once
        if not skip_compute:
            if not skip_fp:
                fingerprint_main(
                    root_dir=root_dir,
                    model_path=model_path,
                    method=run_method,
                    data_modality=data_modality,
                    data_type=data_type,
                    data_space=data_space,
                    same_session=same_session,
                    window_size=window_size,
                    step_size=step_size,
                    segment_size=segment_size,
                    fs=fs,
                    band=band,
                    num_subjects=num_subjects,
                    bootstraps=1,
                    bootstraps_size=None,
                    save_dir=latent_dir,
                    device=device,
                )

            pattern = os.path.join(latent_dir, run_method, f"sub-*_{segment_size}.npy")
            files = sorted(glob(pattern))
            if not files:
                raise RuntimeError(f"No latent vectors found in {pattern}")

            sample_latent = np.load(files[0], mmap_mode="r")
            num_total_windows = sample_latent.shape[1]
            latent_dim = sample_latent.shape[2]
            latent_dtype = sample_latent.dtype
            num_k = max(5, num_total_windows // 10)
            k_vals = np.unique(
                np.logspace(
                    np.log10(2),
                    np.log10(num_total_windows),
                    num=num_k,
                    dtype=int,
                )
            )
            acc_means = []
            acc_stds = []
            step_sizes = []

            for k in tqdm(k_vals, desc=f"Subsampling windows ({method_key})", leave=False):
                idx = np.linspace(0, num_total_windows - 1, k, dtype=int)
                X = np.empty((len(files), 2, latent_dim), dtype=latent_dtype)
                for s_i, fpath in enumerate(files):
                    lat = np.load(fpath, mmap_mode="r")
                    X[s_i] = lat[:, idx, :].mean(axis=1)
                acc_mean, acc_std = bootstrap_metrics(
                    X, bootstraps=bootstraps, bootstraps_size=bootstraps_size
                )
                acc_means.append(acc_mean)
                acc_stds.append(acc_std)
                step_windows = (num_total_windows - 1) / (k - 1)
                step_sizes.append(step_windows * step_size / fs)

            results[method_key] = {
                "k_vals": k_vals,
                "step_sizes": step_sizes,
                "acc_mean": acc_means,
                "acc_std": acc_stds,
            }
            np.save(os.path.join(results_dir, "k_results.npy"), results[method_key])
        else:
            results[method_key] = np.load(
                os.path.join(results_dir, "k_results.npy"), allow_pickle=True
            ).item()

    # Plot results

    colors = {"ae": "C0", "cont": "C1", "cross": "darkgray"}
    labels = {"ae": "Autoencoder", "cont": "Encoder (Cosine Embedding)", "cross": "Encoder (Cross-Entropy)"}

    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, method in enumerate(methods):
        data = results[method]
        label = labels.get(method, method)
        x_vals = np.array(data["step_sizes"]) * fs
        ax.plot(x_vals, data["acc_mean"], color=colors[method], label=label)

    ax.set_xlabel("Step size (samples)")
    ax.set_ylabel("Differentiation accuracy")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(1, x_vals.max())

    # add grid
    ax.grid(alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "fingerprint_windows.svg"))


if __name__ == "__main__":
    main()
