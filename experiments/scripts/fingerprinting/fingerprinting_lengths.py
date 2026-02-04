import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from glob import glob

from fingerprint import main as fingerprint_main
from helpers.fingerprinting_helpers import load_latent_vectors, bootstrap_metrics, compute_num_windows, generate_sliding_windows
from aefp.utils.plotting_utils import style_axes, LABELS
from scipy.stats import ttest_ind
from scipy.stats import t as student_t

ZORDERS = {"ae": 6, "ae_ft": 7}
DEFAULT_ZORDER = 2

def method_zorder(method):
    return ZORDERS.get(method, DEFAULT_ZORDER)


def main():
    
    use_fake_data = False

    skip_compute = True

    dataset = "omega"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}"
    
    if use_fake_data:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fake/lengths/")
    else:
        base_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/fingerprint_lengths/{dataset}/")

    fig_save_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.figures/")
    latent_dir = os.path.join(base_save_dir, "latent_vectors")
    results_dir = os.path.join(base_save_dir, "results")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    data_modality = "meg"
    data_type = "rest"
    data_space = "source_200"
    same_session = False
    band = None
    bootstraps = 100
    bootstraps_size = 0.9
    fs = 150
    # T-test option: use summary stats (mean, std, N) instead of raw samples
    ttest_use_summary = True

    methods = ["ae", "ae_ft", "cont", "aec", "psd"]
    colors = ["C0", "C0", "C1", "C2", "C3"]
    styles = ["-", "--", "-", "-", "-"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_length = 120
    window_size_sec = 6

    # lengths param
    length_step = 1 # in seconds
    seg_lengths = range(window_size_sec, max_length + 1, length_step)

    if not skip_compute:
        # Compute latent vectors once at 120s for each method

        results = {}
        for method in methods:
            if method == "cont":
                model_path = f"/export01/data/{dataset}/saved_models/aefp/encoder/encoder_200.pt"
            else:
                if method == "ae_ft":
                    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200_ft.pt"
                else:
                    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"

            running_avg_X = None
            running_avg_n = 0
            
            acc_alls = []
            acc_means = []
            acc_stds = []
            model, sub_ids = None, None
            for l_i, length_sec in enumerate(tqdm(seg_lengths, desc=f"Processing lengths for {method}")):

                step_size = length_step * fs
                window_size = window_size_sec * fs

                segment_size = (window_size_sec + length_step) * fs
                segment_start = (length_step * l_i) * fs
                if method in {"ae", "ae_ft"}:
                    step_size = 1 # integrate for AE methods
                elif method in {"aec", "psd"}:
                    window_size = length_sec * fs # don't average latent vectors for aec and psd, use max length
                    running_avg_n, running_avg_X = None, None # reset running average
                    segment_size = window_size
                    segment_start = 0

                _, X, n, (model, sub_ids) = fingerprint_main(
                    root_dir=root_dir,
                    model_path=model_path,
                    model=model,
                    sub_ids=sub_ids,
                    method=method,
                    data_modality=data_modality,
                    data_type=data_type,
                    data_space=data_space,
                    same_session=same_session,
                    window_size=window_size,
                    step_size=step_size,
                    segment_size=segment_size,
                    segment_start=segment_start,
                    fs=fs,
                    band=band,
                    num_subjects=num_subjects,
                    bootstraps=1,
                    bootstraps_size=None,
                    save_dir=None,
                    running_avg_X=running_avg_X,
                    running_avg_n=running_avg_n,
                    return_n=True,
                    return_model=True,
                    device=device,
                )
                acc_all = bootstrap_metrics(X, bootstraps=bootstraps, bootstraps_size=bootstraps_size, return_all=True)
                acc_alls.append(acc_all)
                acc_means.append(np.mean(acc_all))
                acc_stds.append(np.std(acc_all))

                running_avg_X = X
                running_avg_n = n

            results[method] = {
                "lengths": seg_lengths,
                "acc_alls": acc_alls,
                "acc_mean": acc_means,
                "acc_std": acc_stds,
            }
            np.save(os.path.join(results_dir, f"{method}_results.npy"), results[method])
            
    else:
        results = {}
        for method in methods:
            results[method] = np.load(os.path.join(results_dir, f"{method}_results.npy"), allow_pickle=True).item()

    # Length-wise one-sided t-tests: AE/AE_FT vs FC (AEC)
    ttest_res = compute_lengthwise_ttests(
        results,
        ae_keys=("ae", "ae_ft"),
        fc_key="aec",
        alpha=0.05,
        use_summary=ttest_use_summary,
        n_bootstraps=bootstraps,
    )
    if ttest_res:
        print("\nOne-sided t-test (AE variants > FC/AEC) by length:")
        for key in ("ae", "ae_ft"):
            if key not in ttest_res:
                continue
            lengths = ttest_res[key]["lengths"]
            sig_mask = ttest_res[key]["significant"]
            sig_lengths = lengths[sig_mask]
            earliest = int(sig_lengths.min()) if sig_lengths.size > 0 else None
            if sig_lengths.size > 0:
                print(f" - {LABELS.get(key, key)} beats FC at lengths (s): {sig_lengths.tolist()} | earliest: {earliest} with p = {ttest_res[key]['pvals'][sig_mask][0]:.4e}")
            else:
                print(f" - {LABELS.get(key, key)}: no significant improvement over FC across lengths")

    # Quick summary printout before plotting
    target_lengths = [10, 30, 60, 120]
    print("Matching accuracies (mean ± sd) at selected lengths:")
    for method in methods:
        data = results[method]
        lengths_arr = np.array(list(data["lengths"])) if isinstance(data["lengths"], range) else np.array(data["lengths"]) 
        acc_means = np.array(data["acc_mean"])
        acc_stds = np.array(data["acc_std"])
        print(f"- {method}:")
        for t in target_lengths:
            if len(lengths_arr) == 0 or t < lengths_arr.min() or t > lengths_arr.max():
                print(f"  {t}s: N/A (out of range)")
                continue
            idxs = np.where(lengths_arr == t)[0]
            if idxs.size == 0:
                idx = int(np.argmin(np.abs(lengths_arr - t)))
            else:
                idx = int(idxs[0])
            print(f"  {t}s: acc={acc_means[idx]:.4f}, sd={acc_stds[idx]:.4f}")

    # Plot results
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for i, (method, data) in enumerate(results.items()):
        lengths = np.array(data["lengths"])
        acc_means = np.array(data["acc_mean"])
        acc_stds = np.array(data["acc_std"])

        mask = lengths >= 8
        lengths = lengths[mask]
        acc_means = acc_means[mask]
        acc_stds = acc_stds[mask]
        base_zorder = method_zorder(method)
        ax.plot(
            lengths,
            acc_means,
            label=LABELS[method],
            color=colors[i],
            linestyle=styles[i],
            linewidth=2,
            zorder=base_zorder,
        )
        ax.fill_between(
            lengths,
            np.array(acc_means) - np.array(acc_stds),
            np.array(acc_means) + np.array(acc_stds),
            alpha=0.2,
            color=colors[i],
            zorder=base_zorder - 0.1,
        )

    if dataset == "camcan":
        crossover = compute_ae_vs_best_ttests(
            results,
            ae_key="ae",
            baseline_keys=("aec", "cont", "psd", "ae_ft"),
            alpha=0.05,
            use_summary=ttest_use_summary,
            n_bootstraps=bootstraps,
        )
        print("crossover", crossover)
        if crossover:
            lengths = np.asarray(crossover["lengths"])
            pvals = np.asarray(crossover["pvals"])
            mean_diffs = np.asarray(crossover["mean_diffs"])
            mask = lengths > 10
            crossover_range = compute_crossover_range(
                lengths[mask],
                mean_diffs[mask],
                pvals[mask],
                alpha=0.05,
            )
            crossover_range = (crossover_range[0]+1, crossover_range[1]+1)  # adjust for initial length offset
            print(crossover_range)
            if crossover_range is not None:
                ax.axvspan(
                    crossover_range[0],
                    crossover_range[1],
                    color="black",
                    alpha=0.2,
                    label="Crossover range",
                    zorder=-20,
                )
                #ax.axvline(crossover_range[0], color="black", linestyle="--", linewidth=1)
                #ax.axvline(crossover_range[1], color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Segment length (s)")
    ax.set_ylabel("Differentiation accuracy")
    ax.legend(loc="center right" if dataset=="omega" else "lower right", bbox_to_anchor=(1, 0.40) if dataset=="omega" else None, frameon=False)
    ax.grid(alpha=0.3)

    #ax.set_ylim(0, None)

    # remove spines top right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    style_axes(ax, n_major=7, n_minor=0)

    if dataset == "camcan":
        ax.set_xticks([int((crossover_range[0] + crossover_range[1])/2), 20, 40, 60, 80, 100, 120])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f"fingerprint_lengths_{dataset}.svg"))


def compute_lengthwise_ttests(
    results,
    ae_keys=("ae", "ae_ft"),
    fc_key="aec",
    alpha=0.05,
    use_summary=False,
    n_bootstraps=None,
):
    """Compute one-sided t-tests per length comparing AE variants to FC (AEC).

    Parameters
    - results: dict of method -> {"lengths", "acc_alls", ...}
    - ae_keys: iterable of method names to compare against FC
    - fc_key: method name used as FC baseline (default "aec")
    - alpha: significance threshold
    - use_summary: if True, use (mean, std, N) instead of raw samples
    - n_bootstraps: number of bootstraps (N) when using summary stats

    Returns
    - dict mapping each ae_key to {"lengths", "pvals", "significant"}
    """
    out = {}
    if fc_key not in results:
        return out

    fc_lengths = results[fc_key]["lengths"]
    lengths_arr = np.array(list(fc_lengths)) if isinstance(fc_lengths, range) else np.array(fc_lengths)
    fc_acc_alls = results[fc_key].get("acc_alls")
    fc_means = results[fc_key].get("acc_mean")
    fc_stds = results[fc_key].get("acc_std")

    for key in ae_keys:
        if key not in results:
            continue
        ae_acc_alls = results[key].get("acc_alls")
        ae_means = results[key].get("acc_mean")
        ae_stds = results[key].get("acc_std")
        pvals = []
        sig = []
        if use_summary:
            if n_bootstraps is None:
                # Try to infer from raw arrays if available
                if ae_acc_alls and fc_acc_alls and len(ae_acc_alls) > 0 and len(fc_acc_alls) > 0:
                    n_bootstraps = min(len(np.asarray(ae_acc_alls[0])), len(np.asarray(fc_acc_alls[0])))
                else:
                    raise ValueError("n_bootstraps must be provided when use_summary=True and raw samples are unavailable.")
            L = min(len(ae_means), len(fc_means))
        else:
            L = min(len(ae_acc_alls), len(fc_acc_alls))
        for i in range(L):
            if use_summary:
                m1 = float(ae_means[i]); s1 = float(ae_stds[i]); n1 = int(n_bootstraps)
                m2 = float(fc_means[i]); s2 = float(fc_stds[i]); n2 = int(n_bootstraps)
                denom = np.sqrt((s1**2)/n1 + (s2**2)/n2)
                if denom == 0:
                    # If no variability, decide by mean difference
                    p = 0.0 if (m1 > m2) else 1.0
                else:
                    t_stat = (m1 - m2) / denom
                    # Welch-Satterthwaite df
                    v1 = (s1**2)/n1; v2 = (s2**2)/n2
                    df_num = (v1 + v2)**2
                    df_den = (v1**2)/(n1 - 1 if n1 > 1 else 1) + (v2**2)/(n2 - 1 if n2 > 1 else 1)
                    df = df_num/df_den if df_den > 0 else max(n1 + n2 - 2, 1)
                    # one-sided p-value: P(T > t_stat)
                    p = student_t.sf(t_stat, df)
            else:
                ae_vals = np.asarray(ae_acc_alls[i])
                fc_vals = np.asarray(fc_acc_alls[i])
                if ae_vals.size == 0 or fc_vals.size == 0:
                    p = np.nan
                else:
                    try:
                        stat, p = ttest_ind(ae_vals, fc_vals, equal_var=False, alternative="greater")
                    except TypeError:
                        # Fallback for SciPy versions without 'alternative'
                        stat, p_two = ttest_ind(ae_vals, fc_vals, equal_var=False)
                        p = p_two / 2.0 if stat > 0 else 1 - (p_two / 2.0)
            pvals.append(p)
            sig.append(np.isfinite(p) and (p < alpha))
        out[key] = {
            "lengths": lengths_arr[:L],
            "pvals": np.asarray(pvals),
            "significant": np.asarray(sig, dtype=bool),
        }
    return out


def compute_ae_vs_best_ttests(
    results,
    ae_key="ae",
    baseline_keys=("aec", "cont", "psd", "ae_ft"),
    alpha=0.05,
    use_summary=False,
    n_bootstraps=None,
):
    """Compare AE to the best non-AE method at each length (one-sided: AE > best)."""
    if ae_key not in results:
        return {}

    ae_lengths = results[ae_key]["lengths"]
    lengths_arr = np.array(list(ae_lengths)) if isinstance(ae_lengths, range) else np.array(ae_lengths)
    ae_means = results[ae_key].get("acc_mean")
    ae_stds = results[ae_key].get("acc_std")
    ae_acc_alls = results[ae_key].get("acc_alls")

    available = [k for k in baseline_keys if k in results and k != ae_key]
    if not available:
        return {}

    pvals = []
    best_keys = []
    mean_diffs = []
    used_lengths = []

    for i, length in enumerate(lengths_arr):
        candidates = [k for k in available if i < len(results[k].get("acc_mean", []))]
        if not candidates:
            break
        best_key = max(candidates, key=lambda k: results[k]["acc_mean"][i])
        best_keys.append(best_key)
        used_lengths.append(length)

        m1 = float(ae_means[i]); s1 = float(ae_stds[i])
        m2 = float(results[best_key]["acc_mean"][i]); s2 = float(results[best_key]["acc_std"][i])
        mean_diffs.append(m1 - m2)

        if use_summary:
            if n_bootstraps is None:
                if ae_acc_alls and results[best_key].get("acc_alls"):
                    n_bootstraps = min(
                        len(np.asarray(ae_acc_alls[i])),
                        len(np.asarray(results[best_key]["acc_alls"][i])),
                    )
                else:
                    raise ValueError("n_bootstraps must be provided when use_summary=True and raw samples are unavailable.")
            n1 = int(n_bootstraps); n2 = int(n_bootstraps)
            denom = np.sqrt((s1**2)/n1 + (s2**2)/n2)
            if denom == 0:
                p = 0.0 if (m1 > m2) else 1.0
            else:
                t_stat = (m1 - m2) / denom
                v1 = (s1**2)/n1; v2 = (s2**2)/n2
                df_num = (v1 + v2)**2
                df_den = (v1**2)/(n1 - 1 if n1 > 1 else 1) + (v2**2)/(n2 - 1 if n2 > 1 else 1)
                df = df_num/df_den if df_den > 0 else max(n1 + n2 - 2, 1)
                p = student_t.sf(t_stat, df)
        else:
            ae_vals = np.asarray(ae_acc_alls[i]) if ae_acc_alls is not None else np.asarray([])
            best_vals = np.asarray(results[best_key].get("acc_alls", [])[i]) if results[best_key].get("acc_alls") else np.asarray([])
            if ae_vals.size == 0 or best_vals.size == 0:
                p = np.nan
            else:
                try:
                    stat, p = ttest_ind(ae_vals, best_vals, equal_var=False, alternative="greater")
                except TypeError:
                    stat, p_two = ttest_ind(ae_vals, best_vals, equal_var=False)
                    p = p_two / 2.0 if stat > 0 else 1 - (p_two / 2.0)
        pvals.append(p)

    return {
        "lengths": np.asarray(used_lengths),
        "pvals": np.asarray(pvals),
        "mean_diffs": np.asarray(mean_diffs),
        "best_keys": best_keys,
        "alpha": alpha,
    }


def compute_crossover_range(lengths, mean_diffs, pvals, alpha=0.05):
    """Find a contiguous range where AE vs best is not significant around the mean crossover."""
    if lengths is None or len(lengths) == 0:
        return None
    lengths = np.asarray(lengths)
    mean_diffs = np.asarray(mean_diffs)
    pvals = np.asarray(pvals)
    pos_idxs = np.where(mean_diffs > 0)[0]
    if pos_idxs.size == 0:
        return None
    cross_idx = int(pos_idxs[0])
    candidate = np.where(pvals >= alpha)[0]
    if candidate.size == 0:
        return (float(lengths[cross_idx]), float(lengths[cross_idx]))

    blocks = []
    start = int(candidate[0])
    prev = int(candidate[0])
    for idx in candidate[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            blocks.append((start, prev))
            start = idx
            prev = idx
    blocks.append((start, prev))

    chosen = None
    for block in blocks:
        if block[0] <= cross_idx <= block[1]:
            chosen = block
            break
    if chosen is None:
        distances = [min(abs(cross_idx - b[0]), abs(cross_idx - b[1])) for b in blocks]
        chosen = blocks[int(np.argmin(distances))]

    return (float(lengths[chosen[0]]), float(lengths[chosen[1]]))


if __name__ == "__main__":
    main()
