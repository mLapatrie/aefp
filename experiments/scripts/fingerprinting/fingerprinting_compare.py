
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from matplotlib import patches as mpatches

from fingerprint import main as fingerprint_main
from helpers.fingerprinting_helpers import load_latent_vectors, bootstrap_metrics
from aefp.utils.plotting_utils import style_axes, LABELS

ZORDERS = {"ae": 6, "ae_ft": 7}
DEFAULT_ZORDER = 2

def method_zorder(method):
    return ZORDERS.get(method, DEFAULT_ZORDER)


def main():
    """Compare two fingerprinting approaches with all methods."""

    use_fake_data = False
    skip_compute = True

    # Dataset can be "camcan" or "omega"
    dataset = "camcan"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}"

    if use_fake_data:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fake/compare/{dataset}/")
    else:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fingerprint_compare/{dataset}/")

    fig_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.figures/")
    os.makedirs(base_save_dir, exist_ok=True)

    data_modality = "meg"
    data_type = "rest"
    data_space = "source_200"
    bootstraps = 100
    bootstraps_size = 0.9
    fs = 150

    segment_size_sec = 120
    step_size_sec = 1

    methods = ["ae", "ae_ft", "cont", "aec", "psd"]
    colors = ["C0", "C0", "C1", "C2", "C3"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == "camcan":
        approaches = { 
            "matched_kernel": {"data_space": "source_200", "same_session": True},
            "same_kernel": {"data_space": "source_200_sk", "same_session": True},
        }
    else:
        approaches = {
            "within": {"data_space": data_space, "same_session": True},
            "between": {"data_space": data_space, "same_session": False},
        }

    results = {}

    if not skip_compute:
        for appr_name, params in approaches.items():

            acc_alls = []
            acc_means = []
            acc_stds = []
            for method in methods:
                print(appr_name, method)

                if method == "cont":
                    model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200.pt"
                else:
                    if method == "ae_ft":
                        model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200_ft.pt"
                    else:
                        model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"

                window_size_sec = 6 if method in {"ae", "ae_ft", "cont"} else segment_size_sec
                step_size = int(step_size_sec * fs) if method == "cont" else 1

                if appr_name in {"matched_kernel", "within"} and method in {"ae", "ae_ft"}:
                    acc_alls.append([1] * bootstraps)
                    acc_means.append(1.0)
                    acc_stds.append(0.0)
                    continue

                _, X = fingerprint_main(
                    root_dir=root_dir,
                    model_path=model_path,
                    method=method,
                    data_modality=data_modality,
                    data_type=data_type,
                    data_space=params["data_space"],
                    same_session=params["same_session"],
                    window_size=window_size_sec * fs,
                    step_size=step_size,
                    segment_size=segment_size_sec * fs,
                    fs=fs,
                    band=None,
                    num_subjects=num_subjects,
                    bootstraps=1,
                    bootstraps_size=None,
                    save_dir=None,
                    device=device,
                )

                acc_all = bootstrap_metrics(X, bootstraps=bootstraps, bootstraps_size=bootstraps_size, return_all=True)
                acc_alls.append(acc_all)
                acc_means.append(np.mean(acc_all))
                acc_stds.append(np.std(acc_all))
            
            results[appr_name] = {
                "methods": methods,
                "acc_alls": acc_alls,
                "acc_mean": acc_means,
                "acc_std": acc_stds,
            }
            np.save(os.path.join(base_save_dir, appr_name, "results.npy"), results[appr_name])

    else:
        for appr_name in approaches:
            results[appr_name] = np.load(os.path.join(base_save_dir, appr_name, "results.npy"), allow_pickle=True).item()

    print(results)

    # Print accuracies and accuracy drops for each method before plotting
    approach_names = list(results.keys())
    print("\nFingerprint accuracies by method:")
    for i, method in enumerate(methods):
        accs = [results[appr]["acc_mean"][i] for appr in approach_names]
        stds = [results[appr]["acc_std"][i] for appr in approach_names]
        acc_str = ", ".join([f"{appr}={accs[j]:.3f}+-{stds[j]:.3f}" for j, appr in enumerate(approach_names)])
        print(f" - {method}: {acc_str}")
        if len(approach_names) >= 2:
            base = accs[0]
            drops = [base - v for v in accs[1:]]
            drop_str = ", ".join([f"{approach_names[j+1]}={drops[j]:.3f}" for j in range(len(drops))])
            print(f"   Drops vs {approach_names[0]}: {drop_str}")

    # Plot results as a slopegraph
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)

    approach_names = list(results.keys())
    x = np.arange(len(approach_names))

    # Plot each method without end labels; use legend instead
    legend_handles = []
    legend_labels = []
    for i, method in enumerate(methods):
        y = [results[appr]["acc_mean"][i] for appr in approach_names]
        yerr = [results[appr]["acc_std"][i] for appr in approach_names]
        linestyle = (0, (4, 4)) if method == "ae_ft" else "-"

        base_zorder = method_zorder(method)
        line, = ax.plot(
            x,
            y,
            color=colors[i],
            marker="o",
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            zorder=base_zorder,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color="black",
            linestyle="none",
            linewidth=2,
            zorder=base_zorder + 0.1,
        )

        # Collect handles and labels for legend (use ae and ae_ft names as-is)
        legend_handles.append(line)
        legend_labels.append(LABELS[method])

    ax.set_xticks(x)
    if dataset == "omega":
        approach_names = ["Within-session", "Between-session"]
    else:
        approach_names = ["Native anatomy", "Surrogate anatomy"]
    ax.set_xticklabels([name.replace("_", " ").capitalize() for name in approach_names])
    ax.set_xlabel("Source reconstruction anatomy" if dataset=="camcan" else "Session type")
    #ax.set_ylabel("Differentiation accuracy")
    if dataset == "camcan":
        ax.set_ylim(0.7, None)
        ax.set_yticks(np.arange(0.7, 1.01, 0.1))

    # Legend showing all methods; ae_ft appears dashed, ae solid
    ax.legend(handles=legend_handles, labels=legend_labels, loc="lower left", frameon=False)

    # remove spines top right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #style_axes(ax, n_major=2, n_minor=0)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"fingerprint_compare_{dataset}.svg"))


if __name__ == "__main__":
    main()
