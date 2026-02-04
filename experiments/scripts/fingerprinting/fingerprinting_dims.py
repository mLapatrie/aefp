import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from fingerprint import main as fingerprint_main
from helpers.fingerprinting_helpers import bootstrap_metrics
from aefp.utils.plotting_utils import style_axes


def main():
    use_fake_data = False

    skip_compute = True

    # dataset: "camcan" (matched vs same kernel) or
    #          "omega" (within vs between session)
    dataset = "omega"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}"

    if use_fake_data:
        base_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.tmp/fake/dims/")
    else:
        base_save_dir = os.path.expanduser(
            f"~/dev/python/aefp/experiments/.tmp/fingerprint_dims/{dataset}/"
        )

    fig_save_dir = os.path.expanduser("~/dev/python/aefp/experiments/.figures/")
    latent_dir = os.path.join(base_save_dir, "latent_vectors")
    results_dir = os.path.join(base_save_dir, "results")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    data_modality = "meg"
    data_type = "rest"
    # data_space is chosen per-approach below
    band = None
    bootstraps = 100
    bootstraps_size = 0.9
    fs = 150
    method = "ae"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segment_size_sec = 120
    segment_size = segment_size_sec * fs
    window_size_sec = 6
    window_size = window_size_sec * fs

    # Subsampled latent dimensionalities to evaluate
    dims = [1024, 2048, 4096, 8192, 16384, 20000]
    dim_samples = 100  # number of random subsets per dimensionality

    # Model path per dataset
    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"

    # Define approaches depending on dataset
    if dataset == "camcan":
        # Matched anatomy (subject-specific kernels) vs Same kernel (shared anatomy)
        approaches = {
            "matched_kernel": {"data_space": "source_200", "same_session": True, "step_size": 150},
            "same_kernel": {"data_space": "source_200_sk", "same_session": True, "step_size": 1},
        }
        approach_order = ["matched_kernel", "same_kernel"]
    elif dataset == "omega":
        # Within-session vs Between-session fingerprinting
        approaches = {
            "within": {"data_space": "source_200", "same_session": True, "step_size": 150},
            "between": {"data_space": "source_200", "same_session": False, "step_size": 1},
        }
        approach_order = ["between", "within"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results_file = os.path.join(results_dir, "results.npy")

    if not skip_compute:
        results = {k: [] for k in approaches.keys()}
        for appr_name, params in approaches.items():
            if dataset == "camcan" and appr_name == "matched_kernel":
                results[appr_name] = [{"dim": dim, "acc_alls": [1], "acc_mean": 1, "acc_std": 0.0} for dim in dims]
                print("Skipping matched_kernel (accuracy=1)")
                continue
            _, X = fingerprint_main(
                root_dir=root_dir,
                model_path=model_path,
                method=method,
                data_modality=data_modality,
                data_type=data_type,
                data_space=params["data_space"],
                same_session=params["same_session"],
                window_size=window_size,
                step_size=params["step_size"],
                segment_size=segment_size,
                fs=fs,
                band=band,
                num_subjects=num_subjects,
                bootstraps=1,
                bootstraps_size=None,
                save_dir=None,
                device=device,
            )

            for dim in dims:
                # Guard against requesting more dims than available
                feat_dim = X.shape[2]
                if dim > feat_dim:
                    # Skip or cap to feat_dim to avoid sampling error
                    # Here we cap to available features.
                    dim_eff = feat_dim
                else:
                    dim_eff = dim

                scores = []
                for _ in range(dim_samples):
                    idx = np.random.choice(X.shape[2], size=dim_eff, replace=False)
                    acc_mean, _ = bootstrap_metrics(
                        X[:, :, idx], bootstraps=bootstraps, bootstraps_size=bootstraps_size
                    )
                    scores.append(acc_mean)

                results[appr_name].append(
                    {"dim": dim, "acc_alls": scores, "acc_mean": np.mean(scores), "acc_std": np.std(scores)}
                )
        np.save(results_file, results)
    else:
        results = np.load(results_file, allow_pickle=True).item()
        # Quick summary printout after loading results
        print("Loaded fingerprint-by-dimension results:")
        for appr in approach_order:
            if appr not in results:
                continue
            rows = results[appr]
            dims_loaded = [r.get("dim") for r in rows]
            means_loaded = [r.get("acc_mean") for r in rows]
            stds_loaded = [r.get("acc_std") for r in rows]
            print(f" - {appr}:")
            for d, m, s in zip(dims_loaded, means_loaded, stds_loaded):
                if d is None:
                    continue
                print(f"   dim={int(d)}: acc={float(m):.4f} ± {float(s):.4f}")
    

    # Extract data for plotting (assumes exactly two approaches per dataset)
    a1, a2 = approach_order
    if dataset == "camcan":
        a2, a1 = a1, a2  # plot same_kernel (a2) as hatched bars
        l2 = "Native anatomy"
        l1 = "Surrogate anatomy"
    elif dataset == "omega":
        l2 = "Within-session"
        l1 = "Between-session"
    
    print(results)
    dims = [r["dim"] for r in results[a1]]
    a1_means = [r["acc_mean"] for r in results[a1]]
    a1_stds = [r["acc_std"] for r in results[a1]]
    a2_means = [r["acc_mean"] for r in results[a2]]
    a2_stds = [r["acc_std"] for r in results[a2]]

    # Print accuracy per dimensionality before plotting
    print("Accuracy by dimensionality:")
    print(f"dim\t{l1}_mean±std\t{l2}_mean±std")
    for i, dim in enumerate(dims):
        print(
            f"{dim}\t{a1_means[i]:.4f}±{a1_stds[i]:.4f}\t{a2_means[i]:.4f}±{a2_stds[i]:.4f}"
        )

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_axisbelow(True)
    ax.bar(x - width / 2, a1_means, width, yerr=a1_stds, label=l1)
    ax.bar(
        x + width / 2,
        a2_means,
        width,
        yerr=a2_stds,
        fill=True,
        facecolor="darkgrey",
        #edgecolor="black",
        #hatch="//",
        label=l2,
    )
    #ax.plot(x - width / 2, a1_means, color="C0")
    #ax.plot(x + width / 2, a2_means, color="C1", linestyle="--")
    ax.set_xlabel("Latent dimensionality")
    ax.set_ylabel("Differentiation accuracy")
    ax.legend(loc="lower left", framealpha=1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    # remove spines top right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #style_axes(ax, n_major=6, n_minor=0)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"fingerprint_dims_{dataset}.svg"))


if __name__ == "__main__":
    main()
