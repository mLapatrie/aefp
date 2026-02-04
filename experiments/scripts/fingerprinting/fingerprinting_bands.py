import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from fingerprint import main as fingerprint_main
from helpers.fingerprinting_helpers import (
    BANDS_DEF,
    load_latent_vectors,
    bootstrap_metrics,
)
from aefp.utils.plotting_utils import style_axes, LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ZORDERS = {"ae": 6, "ae_ft": 7}
DEFAULT_ZORDER = 2
BOX_ZORDER_OFFSET = 1

def method_zorder(method):
    return ZORDERS.get(method, DEFAULT_ZORDER)

def main():

    use_fake_data = False

    skip_compute = True

    dataset = "omega"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}/"

    if use_fake_data:
        base_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.tmp/fake/bands/")
    else:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fingerprint_bands/{dataset}/")

    fig_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.figures/")
    latent_dir = os.path.join(base_save_dir, "latent_vectors")
    results_dir = os.path.join(base_save_dir, "results")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    data_modality = "meg"
    data_type = "rest"
    data_space = "source_200"
    same_session = False
    bootstraps = 100
    bootstraps_size = 0.9
    fs = 150

    segment_size_sec = 120
    step_size_sec = 1

    if dataset == "camcan":
        methods = ["ae", "cont", "aec", "psd"]
        colors = ["C0", "C1", "C2", "C3"]
    else:
        methods = ["ae", "ae_ft", "cont", "aec", "psd"]
        colors = ["C0", "C0", "C1", "C2", "C3"]
    bands = list(BANDS_DEF.keys())

    # Compute latent vectors once for each band and method
    if not skip_compute:
        results = {}
        for method in methods:
            if method == "cont":
                model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200.pt"
            else:
                if method == "ae_ft":
                    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200_ft.pt"
                else:
                    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"

            window_size_sec = 6
            step_size = 1
            if method in {"cont"}:
                window_size_sec = 6
                step_size = int(step_size_sec * fs)
            elif method in {"aec", "psd"}:
                window_size_sec = segment_size_sec

            acc_means = []
            acc_stds = []
            accs = []
            for band in tqdm(bands, leave=False, desc=f"Bands ({method})"):
                _, X = fingerprint_main(
                    root_dir=root_dir,
                    model_path=model_path,
                    method=method,
                    data_modality=data_modality,
                    data_type=data_type,
                    data_space=data_space,
                    same_session=same_session,
                    window_size=window_size_sec * fs,
                    step_size=step_size,
                    segment_size=segment_size_sec * fs,
                    fs=fs,
                    band=band,
                    num_subjects=num_subjects,
                    bootstraps=1,
                    bootstraps_size=None,
                    save_dir=None,
                    device=device,
                )

                values = bootstrap_metrics(
                    X,
                    bootstraps=bootstraps,
                    bootstraps_size=bootstraps_size,
                    return_all=True,
                )
                print(values.mean(), values.std())
                acc_means.append(values.mean())
                acc_stds.append(values.std())
                accs.append(values)
            
            results[method] = {
                "bands": bands,
                "acc_mean": acc_means,
                "acc_std": acc_stds,
                "accs": accs,
            }
            np.save(os.path.join(results_dir, f"{method}_results.npy"), results[method])

    else:
        results = {}
        for method in methods:
            results[method] = np.load(
                os.path.join(results_dir, f"{method}_results_fixed.npy"), allow_pickle=True
            ).item()
        # Quick summary printout after loading results
        print("Loaded fingerprint-by-band results:")
        for method in methods:
            if method not in results:
                continue
            data = results[method]
            bands_loaded = data.get("bands", [])
            means_loaded = data.get("acc_mean", [])
            stds_loaded = data.get("acc_std", [])
            print(f" - {LABELS.get(method, method)}:")
            for b, m, s in zip(bands_loaded, means_loaded, stds_loaded):
                try:
                    print(f"   {b}: acc={float(m):.4f} ± {float(s):.4f}")
                except (TypeError, ValueError):
                    # In case of missing or malformed entries
                    print(f"   {b}: acc=N/A")

    # Plot results as violin plots
    exclude_idx = 1
    bands_filtered = bands#[b for i, b in enumerate(bands) if i != exclude_idx]

    spacing = 3   # spacing between bands
    width = 0.3   # box width
    x = np.arange(len(bands_filtered)) * spacing
    n_methods = len(results)
    offset = (n_methods - 1) / 2

    fig, ax = plt.subplots(figsize=(5.5, 4))

    method_positions = {}  # store x,y for connecting lines

    for i, (method, data) in enumerate(results.items()):
        accs_filtered = data["accs"]#[acc for j, acc in enumerate(data["accs"]) if j != exclude_idx]
        
        pos = x + (i - offset) * width
        
        medians = []
        base_zorder = method_zorder(method)
        box_zorder = base_zorder + BOX_ZORDER_OFFSET
        
        # Box plot for each band
        for band_idx, vals in enumerate(accs_filtered):
            bp = ax.boxplot(vals, positions=[pos[band_idx]], widths=width,
                            patch_artist=True, showcaps=True, showfliers=False,
                            zorder=box_zorder)
            
            # Style
            for box in bp["boxes"]:
                box.set_zorder(box_zorder)
                if i == 1 and dataset == "omega":  # ae_ft
                    box.set(facecolor="C0", alpha=1, edgecolor="black", hatch="///")
                else:
                    box.set(facecolor=colors[i], alpha=1, edgecolor="black")
            for median in bp["medians"]:
                median.set_zorder(box_zorder + 0.2)
                median.set(color="black", linewidth=1.2)
            for whisker in bp["whiskers"]:
                whisker.set_zorder(box_zorder + 0.1)
                whisker.set(color="black", linewidth=1)
            for cap in bp["caps"]:
                cap.set_zorder(box_zorder + 0.1)
                cap.set(color="black", linewidth=1)
            
            # Store medians for line connection
            med_val = np.median(vals)
            medians.append(med_val)
        
        method_positions[method] = (pos, medians)

        linestyle = (0, (4, 4)) if method == "ae_ft" else "-"
        ax.plot([], [], color=colors[i], label=LABELS[method], markersize=6, marker="o", linestyle=linestyle, linewidth=2)  # legend proxy

    # Add dashed lines connecting medians
    for method, (pos, meds) in method_positions.items():
        linestyle = "--" if method == "ae_ft" else "-"
        ax.plot(
            pos,
            meds,
            linestyle=linestyle,
            color=colors[list(results.keys()).index(method)],
            alpha=1,
            linewidth=2,
            zorder=method_zorder(method),
        )


    # remove spines top right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(linestyle='-', axis='x', alpha=0.3)
    ax.grid(linestyle='-', axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(bands_filtered)
    ax.set_xlabel("Frequency band")
    #ax.set_ylabel("Differentiation accuracy")
    ax.set_ylim(None, None)
    ax.legend(loc="center right" if dataset=="omega" else "lower left", bbox_to_anchor=(1.02, 0.38) if dataset=="omega" else None, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"fingerprint_bands_{dataset}.svg"))


if __name__ == "__main__":
    main()
