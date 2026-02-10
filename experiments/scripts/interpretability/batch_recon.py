import argparse
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch
import networkx as nx
from mne.time_frequency import tfr_array_morlet
from mne.baseline import rescale

from scipy.stats import t as student_t
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import shutil

from aefp.datasets.meg_dataset import MEGDataset
from aefp.utils.meg_utils import get_network_indices, compute_psd_torch, compute_aec_torch
from aefp.utils.utils import load_autoencoder
from aefp.utils.fingerprinting_utils import get_valid_test_subjects
from aefp.utils.plotting_utils import style_axes

from experiments.scripts.fingerprinting.helpers.fingerprinting_helpers import BANDS_DEF
from scipy.stats import pearsonr


FS_DEFAULT = 150

device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
fig_path = os.path.expanduser("~/dev/python/aefp/experiments/.figures/interpretability/reconstruction/")
os.makedirs(fig_path, exist_ok=True)

seed = 3


def main():
    dataset = "omega"
    gen = False
    use_mse = False  # set True to report normalized MSE metrics instead of correlations
    to_plot = "psd"  # ""psd", "fc", "tfr"
    plot_labels = False

    data_path = f"/export01/data/{dataset}/"
    
    if dataset == "camcan":
        model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"
    else:
        model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200_1.pt"

    out_path = os.path.expanduser(f"~/dev/python/aefp/experiments/.figures/interpretability/{dataset}/reconstruction/")
    os.makedirs(out_path, exist_ok=True)

    anatomy_path = None
    
    n_subjects = 1
    
    fs = 150
    fs_cutoff = 50
    avg_fc_networks = False
    n_windows = 10 # 60s

    torch.manual_seed(seed)
    np.random.seed(seed)

    real, rec = load_real_and_recon(
        model_path=model_path,
        data_path=data_path,
        n_subjects=n_subjects,
        block_shape=(200, 900),
        n_windows=n_windows,
        gen=gen,
    )

    print(real.shape, rec.shape)

    #compute_and_print_recon_metrics(
    #    real=real,
    #    rec=rec,
    #    dataset=dataset,
    #    fs=fs,
    #    metric=("mse" if use_mse else "corr"),
    #)

    if to_plot == "fc":
        plot_fc_graphs(real, rec, dataset=dataset, anatomy_path=anatomy_path, avg_networks=avg_fc_networks)
    elif to_plot == "psd":
        plot_psd(real, rec, dataset=dataset, anatomy_path=anatomy_path, fs=fs, fs_cutoff=fs_cutoff, plot_labels=plot_labels)
    elif to_plot == "tfr":
        plot_tfr_comparison(
            real,
            rec,
            dataset=dataset,
            anatomy_path=anatomy_path,
            sfreq=fs,
            freqs=np.arange(1, 31),
            n_cycles=3,
            use_coi=False,
            plot_labels=plot_labels,
        )



def load_real_and_recon(model_path, data_path, n_subjects=1,
                            block_shape=(200, 900), n_windows: int = 1, gen: bool = False):
    """Load real windows and corresponding reconstructions.

    Parameters
    ----------
    model_path : str
        Path to trained autoencoder.
    data_path : str
        Path to dataset root.
    n_subjects : int
        Number of subjects to load.
    block_shape : tuple
        Shape of a single time window (channels, timepoints).
    n_windows : int
        Number of consecutive windows to load per subject.

    Returns
    -------
    real : np.ndarray
        Array of shape (n_subjects, n_windows, channels, timepoints) containing
        real MEG windows.
    gen : np.ndarray
        Reconstructions with the same shape as ``real``.
    """

    model, _, sub_dict = load_autoencoder(model_path)
    model.to(device).eval()

    test_subjects, _ = get_valid_test_subjects(sub_dict, data_path, same_session=True)
    # shuffle and select n_subjects
    np.random.shuffle(test_subjects)
    test_subjects = test_subjects[:n_subjects]

    dataset = MEGDataset(
        data_path=data_path,
        block_shape=block_shape,
        max_num_blocks=None, # load entire recording
        sub_ids=test_subjects,
        device=device,
    )

    real = np.stack([
        np.stack([
            dataset.data[i][0][j].cpu().numpy() for j in range(min(n_windows, len(dataset.data[i][0])))
        ])
        for i in range(len(dataset.data))
    ])

    with torch.no_grad():
        # Flatten subjects and windows into a single batch axis
        flat = real.reshape(-1, real.shape[-2], real.shape[-1])  # (N, C, T)
        tensor = torch.tensor(flat, dtype=torch.float32, device=device)

        # Process encoding/decoding in batches of 16 to limit memory usage
        batch_size = 8
        outs = []
        n_total = tensor.shape[0]

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = tensor[start:end].unsqueeze(1)  # (B, 1, C, T)

            if gen:
                # Sample a random latent with the same shape as the encoder sample
                z_sample = model.encode(batch).sample()
                rand_z = torch.randn_like(z_sample)
                dec_b = model.decode(rand_z)
            else:
                dec_b, _ = model(batch)

            outs.append(dec_b.squeeze(1))  # (B, C, T)

        dec = torch.cat(outs, dim=0)  # (N, C, T)
        rec = dec.cpu().numpy().reshape(real.shape)
        if gen:
            return None, rec
    return real, rec


def _psd_avg(data, fs, fs_cutoff=50):
    psd, freqs = compute_psd_torch(torch.tensor(data).unsqueeze(0), fs=fs, log=True)
    psd = psd.squeeze(0).numpy()  # shape (n_channels, n_freqs)
    freqs = freqs.numpy()  # shape (n_freqs,)
    num_idx = np.where(freqs <= fs_cutoff)[0][-1] + 1
    psd = psd[:, :num_idx]  # keep only frequencies up to fs
    freqs = freqs[:num_idx]  # keep corresponding frequencies

    return freqs, np.mean(psd, axis=0), np.std(psd, axis=0)


def get_t_stat(mean1, std1, n1, mean2, std2, n2):
    # t‐statistic
    se = np.sqrt(std1**2/n1 + std2**2/n2)
    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite df
    df_num = (std1**2/n1 + std2**2/n2)**2
    df_den = ( (std1**2/n1)**2 / (n1 - 1)
             + (std2**2/n2)**2 / (n2 - 1) )
    df = df_num / df_den

    return t_stat, df


def get_pval(mean1, std1, n1, mean2, std2, n2):
    t_stat, df = get_t_stat(mean1, std1, n1, mean2, std2, n2)
    p_val = 2 * student_t.sf(np.abs(t_stat), df)
    return t_stat, df, p_val


def bayes_factor01_bic(t_stat, df, n_total):
    # BIC approximation to the Bayes factor for the point null.
    return np.sqrt(n_total) * (1.0 + (t_stat**2) / df) ** (-0.5 * n_total)


def _flatten_upper_tri(mat: np.ndarray) -> np.ndarray:
    """Return flattened upper triangle (excluding diagonal)."""
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _bandpass_np(x_np: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 3) -> np.ndarray:
    """Simple Butterworth bandpass on shape (C, T)."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    x_np = x_np.astype(np.float64)
    return filtfilt(b, a, x_np, axis=1)


def _peak_alpha_topography(data: np.ndarray, fs: int = FS_DEFAULT, alpha_band=(8, 12)) -> np.ndarray:
    """Compute per-parcel peak alpha frequency topography for a batch.

    Accepts shapes (subjects, windows, channels, timepoints) or (windows, channels, timepoints).
    Returns a 1D array of length `channels` with peak alpha frequency per parcel (averaged over windows).
    """
    if data is None:
        raise ValueError("data must be provided for peak alpha computation")
    if data.ndim == 4:
        # use first subject for a consistent topography as in plotting code
        data = data[0]
    # Average PSD over windows, then take alpha peak per parcel
    psds = []
    freqs_all = None
    for win in data:
        psd, freqs = compute_psd_torch(torch.tensor(win).unsqueeze(0), fs=fs, log=False)
        psds.append(psd.squeeze(0).numpy())  # (C, F)
        freqs_all = freqs.numpy()
    psd_mean = np.mean(psds, axis=0)  # (C, F)
    mask = (freqs_all >= alpha_band[0]) & (freqs_all <= alpha_band[1])
    alpha_psd = psd_mean[:, mask]
    alpha_freqs = freqs_all[mask]
    peak_idx = np.argmax(alpha_psd, axis=1)
    peak_freqs = alpha_freqs[peak_idx]  # (C,)
    return peak_freqs


def compute_and_print_recon_metrics(real: np.ndarray,
                                    rec: np.ndarray,
                                    dataset: str,
                                    fs: int = FS_DEFAULT,
                                    alpha_band=(8, 13),
                                    metric: str = "corr") -> None:
    """Compute and print reconstruction metrics for a batch of reconstructions.

    Metrics printed:
    - Mean channel-wise time-series reconstruction accuracy: Pearson r (and %) or MSE (after z-scoring by real stats).
    - Correlation between channel amplitude (real) and reconstruction accuracy (uses corr(amp, -mse) for MSE).
    - AEC reconstruction: broadband and per-band; gamma drop in pp (corr) or gamma MSE increase (mse).
    - PSD difference summary: fraction of non-significant freqs ≤ 30 Hz.
    - Peak-alpha-frequency topography: corr or MSE (z-scored).

    Expects shapes (subjects, windows, channels, timepoints).
    """
    assert metric in {"corr", "mse"}, "metric must be 'corr' or 'mse'"
    if real is None:
        print("No real data provided; cannot compute reconstruction metrics.")
        return

    if real.ndim != 4 or rec.ndim != 4:
        raise ValueError("Expected real and rec with shape (S, W, C, T)")

    S, W, C, T = real.shape

    # Utility: z-score an array using reference stats
    def _zscore_with_ref(x: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu = ref.mean()
        sd = ref.std()
        if not np.isfinite(sd) or sd < eps:
            sd = eps
        return (x - mu) / sd

    # 1) Channel-wise reconstruction metric computed per subject, then summarize across subjects
    subj_vals = []  # subject-level mean across channels
    subj_amp_coupling = []  # correlation between amplitude and accuracy per subject
    for s in range(S):
        vals_per_ch = []
        amp_per_ch = []
        for c in range(C):
            real_flat = real[s, :, c, :].reshape(-1)
            rec_flat = rec[s, :, c, :].reshape(-1)
            if metric == "corr":
                if np.std(real_flat) == 0 or np.std(rec_flat) == 0:
                    v = np.nan
                else:
                    v = np.corrcoef(real_flat, rec_flat)[0, 1]
            else:
                rz = _zscore_with_ref(real_flat, real_flat)
                gz = _zscore_with_ref(rec_flat, real_flat)
                v = float(np.mean((rz - gz) ** 2))
            vals_per_ch.append(v)
            amp_per_ch.append(np.std(real_flat))
        vals_per_ch = np.array(vals_per_ch, float)
        amp_per_ch = np.array(amp_per_ch, float)
        with np.errstate(invalid='ignore'):
            subj_vals.append(np.nanmean(vals_per_ch))
            if np.sum(~np.isnan(vals_per_ch)) > 2:
                if metric == "corr":
                    subj_amp_coupling.append(pearsonr(amp_per_ch[~np.isnan(vals_per_ch)],
                                                       vals_per_ch[~np.isnan(vals_per_ch)])[0])
                else:
                    subj_amp_coupling.append(pearsonr(amp_per_ch[~np.isnan(vals_per_ch)],
                                                       (-vals_per_ch)[~np.isnan(vals_per_ch)])[0])
            else:
                subj_amp_coupling.append(np.nan)

    subj_vals = np.array(subj_vals, float)
    subj_amp_coupling = np.array(subj_amp_coupling, float)
    mean_val, sd_val = float(np.nanmean(subj_vals)), float(np.nanstd(subj_vals))
    if metric == "corr":
        print(f"Across subjects: reconstruction accuracy (Pearson r) = {mean_val:.3f} ± {sd_val:.3f} ({mean_val*100:.1f}% ± {sd_val*100:.1f}%)")
        if np.sum(~np.isnan(subj_amp_coupling)) > 0:
            print(f"Amplitude–accuracy coupling (per subject): corr(amp, r) = {np.nanmean(subj_amp_coupling):.3f} ± {np.nanstd(subj_amp_coupling):.3f}")
    else:
        print(f"Across subjects: reconstruction error (MSE, z-scored) = {mean_val:.4f} ± {sd_val:.4f}")
        if np.sum(~np.isnan(subj_amp_coupling)) > 0:
            print(f"Amplitude–accuracy coupling (per subject): corr(amp, -mse) = {np.nanmean(subj_amp_coupling):.3f} ± {np.nanstd(subj_amp_coupling):.3f}")

    # 2) AEC reconstruction accuracy (broadband and per band)
    def aec_metric_for_batch(x_real: np.ndarray, x_rec: np.ndarray) -> float:
        # Compute per-sample AEC correlation or MSE and average
        x_r = torch.tensor(x_real.reshape(-1, C, T))
        x_g = torch.tensor(x_rec.reshape(-1, C, T))
        aec_r = compute_aec_torch(x_r).numpy()
        aec_g = compute_aec_torch(x_g).numpy()
        vals = []
        for i in range(aec_r.shape[0]):
            v1 = _flatten_upper_tri(aec_r[i])
            v2 = _flatten_upper_tri(aec_g[i])
            if metric == "corr":
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                vals.append(np.corrcoef(v1, v2)[0, 1])
            else:
                v1z = _zscore_with_ref(v1, v1)
                v2z = _zscore_with_ref(v2, v1)
                vals.append(float(np.mean((v1z - v2z) ** 2)))
        return float(np.mean(vals)) if vals else float('nan')

    # AEC per subject, then summarize mean ± SD
    aec_vals = []
    for s in range(S):
        aec_vals.append(aec_metric_for_batch(real[s:s+1], rec[s:s+1]))
    aec_vals = np.array(aec_vals, float)
    aec_mean, aec_sd = float(np.nanmean(aec_vals)), float(np.nanstd(aec_vals))
    if metric == "corr":
        print(f"Broadband AEC reconstruction accuracy = {aec_mean:.3f} ± {aec_sd:.3f} ({aec_mean*100:.1f}% ± {aec_sd*100:.1f}%)")
    else:
        print(f"Broadband AEC reconstruction error (MSE, z-scored) = {aec_mean:.4f} ± {aec_sd:.4f}")

    band_accs = {}
    for band_name, (lo, hi) in BANDS_DEF.items():
        # Skip "broadband" here to avoid duplication
        if band_name == "broadband":
            continue
        # Bandpass per sample, compute AEC metric
        x_real_bp = np.array([_bandpass_np(x, lo, hi, fs) for x in real.reshape(-1, C, T)])
        x_rec_bp  = np.array([_bandpass_np(x, lo, hi, fs) for x in rec.reshape(-1, C, T)])
        # Compute per subject values
        subj_band_vals = []
        x_real_bp = x_real_bp.reshape(S, W, C, T)
        x_rec_bp  = x_rec_bp.reshape(S, W, C, T)
        for s in range(S):
            subj_band_vals.append(aec_metric_for_batch(x_real_bp[s:s+1], x_rec_bp[s:s+1]))
        subj_band_vals = np.array(subj_band_vals, float)
        band_accs[band_name] = (float(np.nanmean(subj_band_vals)), float(np.nanstd(subj_band_vals)))
    if band_accs:
        if metric == "corr":
            # Print summary and gamma drop
            print("AEC reconstruction accuracy by band (mean ± SD):")
            ordered = [b for b in ["delta","theta","alpha","beta","gamma"] if b in band_accs]
            for k in ordered:
                m, sd = band_accs[k]
                print(f" - {k:>5}: {m:.3f} ± {sd:.3f} ({m*100:.1f}% ± {sd*100:.1f}%)")
            if "gamma" in band_accs and ordered:
                others = [band_accs[k][0] for k in ordered if k != "gamma"]
                if others:
                    drop_pp = (np.mean(others) - band_accs["gamma"][0]) * 100.0
                    print(f"Gamma drop ≈ {drop_pp:.1f} pp relative to other bands (mean)")
        else:
            print("AEC reconstruction error (MSE, z-scored) by band (mean ± SD):")
            ordered = [b for b in ["delta","theta","alpha","beta","gamma"] if b in band_accs]
            for k in ordered:
                m, sd = band_accs[k]
                print(f" - {k:>5}: {m:.4f} ± {sd:.4f}")
            if "gamma" in band_accs and ordered:
                others = [band_accs[k][0] for k in ordered if k != "gamma"]
                if others:
                    inc = band_accs["gamma"][0] - np.mean(others)
                    print(f"Gamma MSE increase ≈ {inc:.4f} relative to other bands (mean)")

    # 3) PSD difference summary (Welch’s test per frequency)
    real_psd, freqs = compute_psd_torch(torch.tensor(real.reshape(-1, C, T)), fs=fs, log=True)
    rec_psd,  _     = compute_psd_torch(torch.tensor(rec.reshape(-1, C, T)),  fs=fs, log=True)
    real_psd = real_psd.numpy()  # (N,C,F)
    rec_psd  = rec_psd.numpy()
    # Aggregate across channels and windows as samples per frequency
    real_flat = real_psd.reshape(-1, real_psd.shape[-1])  # (N*C, F)
    rec_flat  =  rec_psd.reshape(-1,  rec_psd.shape[-1])  # (N*C, F)
    pvals = []
    for i in range(real_flat.shape[1]):
        m1, s1 = np.mean(real_flat[:, i]), np.std(real_flat[:, i])
        m2, s2 = np.mean(rec_flat[:, i]),  np.std(rec_flat[:, i])
        _, _, p = get_pval(m1, s1, real_flat.shape[0], m2, s2, rec_flat.shape[0])
        pvals.append(p)
    pvals = np.asarray(pvals)
    below_30 = freqs.numpy() <= 30
    frac_nonsig = np.mean(pvals[below_30] >= 0.05)
    print(f"PSD: {frac_nonsig*100:.1f}% of frequencies ≤ 30 Hz show no significant difference (p ≥ 0.05)")

    # 4) Peak alpha frequency topography metric
    # Peak alpha frequency topography: compute per subject and summarize
    paf_vals = []
    for s in range(S):
        real_peaks = _peak_alpha_topography(real[s], fs=fs, alpha_band=alpha_band)
        rec_peaks  = _peak_alpha_topography(rec[s],  fs=fs, alpha_band=alpha_band)
        if metric == "corr":
            if np.std(real_peaks) == 0 or np.std(rec_peaks) == 0:
                paf_vals.append(np.nan)
            else:
                paf_vals.append(np.corrcoef(real_peaks, rec_peaks)[0, 1])
        else:
            rpz = _zscore_with_ref(real_peaks, real_peaks)
            gpz = _zscore_with_ref(rec_peaks, real_peaks)
            paf_vals.append(float(np.mean((rpz - gpz) ** 2)))
    paf_vals = np.array(paf_vals, float)
    if metric == "corr":
        print(f"Peak-alpha-frequency topography correlation = {np.nanmean(paf_vals):.3f} ± {np.nanstd(paf_vals):.3f} ({np.nanmean(paf_vals)*100:.1f}% ± {np.nanstd(paf_vals)*100:.1f}%)")
    else:
        print(f"Peak-alpha-frequency topography error (MSE, z-scored) = {np.nanmean(paf_vals):.4f} ± {np.nanstd(paf_vals):.4f}")


def plot_psd(real_data, rec_data, dataset, anatomy_path=None,
             fs=FS_DEFAULT, fs_cutoff=50, plot_labels=True):
    """
    Plot PSD for selected network(s) comparing real vs. generated data.

    Parameters
    ----------
    real_data, rec_data : np.ndarray
        Arrays of shape (batch, channels, timepoints) or
        (batch, windows, channels, timepoints). If multiple windows are
        provided, the first window is used.
    """

    plot_real = type(real_data) is np.ndarray

    real_data = real_data[:, 0] if plot_real else None
    rec_data = rec_data[:, 0]
    # 1) get mapping name → channel-indices
    network_indices = get_network_indices(dataset, anatomy_path)
    items = list(network_indices.items())  # [(name1, idxs1), (name2, idxs2), …]

    # add whole brain as last entry
    all_idxs = np.arange(rec_data.shape[1])
    network_indices["Whole"] = all_idxs

    # 3) make 2-row figure
    fig, axes = plt.subplots(
        2, 9, # all 17 networks + whole brain
        figsize=(20, 4),
        gridspec_kw={"wspace": 0, "hspace": 0}
    )
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4) loop over each selected network
    vmin = np.inf
    vmax = -np.inf
    net_sigs = []
    for i, (net_name, idxs) in enumerate(network_indices.items()):

        row_idx = 0 if i < 9 else 1
        col_idx = i if i < 9 else i - 9
        ax = axes[row_idx, col_idx]

        # subset your data to just that network’s channels
        real_sub = real_data[0, idxs, :] if plot_real else None  # shape (n_ch_net, n_time)
        gen_sub  = rec_data[0,  idxs, :]

        # compute PSD + std
        _, psd_r, std_r = _psd_avg(real_sub, fs, fs_cutoff=fs_cutoff) if plot_real else (None, None, None)
        freqs, psd_g, std_g = _psd_avg(gen_sub,  fs, fs_cutoff=fs_cutoff)

        if not plot_real:
            psd_r = psd_g
            std_r = std_g

        # update vmin/vmax
        vmin = min(vmin, (psd_r - std_r).min(), (psd_g - std_g).min())
        vmax = max(vmax, (psd_r + std_r).max(), (psd_g + std_g).max())

        # plot real (solid) + generated (dashed)
        if plot_real:
            ax.plot(freqs, psd_r,       label=f"{net_name} Real", color="black")
            ax.fill_between(freqs, psd_r-std_r, psd_r+std_r,
                                alpha=0.2, label="_nolegend_", color="black")
        ax.plot(freqs, psd_g, linestyle="-", label=f"{net_name} Gen", color="red")
        ax.fill_between(freqs, psd_g-std_g, psd_g+std_g,
                             alpha=0.2, label="_nolegend_", color="red")
        #ax.set_title(net_name, fontsize=11)
        # put title as text in middle top of plot
        ax.text(0.5, 0.9, net_name, fontsize=14,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

        if plot_real:
            # compute pointwise Bayes factors (null vs alternative) with correction
            t_stats = []
            dfs = []
            for i in range(len(freqs)):
                t_stat, df = get_t_stat(
                    psd_r[i], std_r[i], real_sub.shape[0],
                    psd_g[i], std_g[i],  gen_sub.shape[0]
                )
                t_stats.append(t_stat)
                dfs.append(df)
            t_stats = np.array(t_stats)
            dfs = np.array(dfs)

            bf01 = bayes_factor01_bic(t_stats, dfs, real_sub.shape[0] + gen_sub.shape[0])
            support_null = bf01 >= 3.0
            net_sigs.append(support_null)

    # set shared x/y limits
    for ax_i, ax in enumerate(axes.flatten()):
        ax.set_xlim(0, fs_cutoff)
        ax.set_ylim(vmin - 0.02 * (vmax - vmin), vmax + 0.02 * (vmax - vmin))
        
        if ax_i != 9 or not plot_labels:
            ax.set_xticks([])
            ax.set_yticks([])

    if plot_labels:
        axes[1, 0].set_ylabel("PSD (AU²/Hz)")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        #ax.legend(loc="upper right")

    # add sig line
    for i, sig in enumerate(net_sigs):
        axes_flatten = axes.flatten()
        ax = axes_flatten[i]
        sig = net_sigs[i]
        ax.plot(freqs[sig], np.ones(sig.sum()) * (ax.get_ylim()[0] + 0.75),
                marker="o", linestyle="None", color="black", markersize=3,)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # remove spines
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    #plt.tight_layout()
    os.makedirs(f"{fig_path}/{dataset}", exist_ok=True)
    plt.savefig(os.path.join(fig_path, f"{dataset}/{"reconstruction" if plot_real else "generated"}_batch_psd_s{seed}.svg"))



def plot_fc_graphs(real_data, rec_data, dataset, anatomy_path=None, avg_networks=False, plot_labels=False):
    """
    Plot a batch of FC graphs by frequency band.

    Layout: 2x6 (Real vs. Reconstructed rows; 6 bands columns).
    If ``real_data`` is None (gen=True), plot a single row (1x6) for generated.
    """

    # Determine whether to plot real row
    plot_real = isinstance(real_data, np.ndarray)

    # Use first subject/window as in plot_psd
    real_win = real_data[:, 0] if plot_real else None  # (B,C,T) -> take first later
    rec_win = rec_data[:, 0]

    # Network indices for optional averaging / ordering
    network_indices = get_network_indices(dataset, anatomy_path)
    names, idx_lists = zip(*network_indices.items())

    # Helper: bandpass filter (SciPy) applied on numpy array shaped (C,T)
    def _butter_bandpass_np(x_np: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 3) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        x_np = x_np.astype(np.float64)
        return filtfilt(b, a, x_np, axis=1)

    # Build band-ordered list
    band_items = list(BANDS_DEF.items())

    # Pre-compute AEC matrices per band
    def compute_band_aec(x_np: np.ndarray, fs: int = FS_DEFAULT) -> list:
        mats = []
        for _, (lo, hi) in band_items:
            xf = torch.tensor(np.array(_butter_bandpass_np(x_np, lo, hi, fs)))
            mat = compute_aec_torch(xf.unsqueeze(0)).squeeze(0).numpy()
            mats.append(mat)
        return mats  # list of (P,P)

    # Take first subject in batch to match plot_psd behavior
    gen_mats = compute_band_aec(rec_win[0])  # list of (P,P)
    real_mats = compute_band_aec(real_win[0]) if plot_real else None

    # Reorder channel-level FC matrices so channels are grouped by network
    # (first network's channels first, then second, etc.). Also update
    # idx_lists to reflect contiguous blocks after reordering.
    if not avg_networks:
        # Build concatenated ordering of channels by network
        per_net_sorted = [np.array(sorted(list(idxs))) for idxs in idx_lists]
        order = np.concatenate(per_net_sorted).astype(int)

        # Apply permutation to all FC mats (rows and cols)
        gen_mats = [m[np.ix_(order, order)] for m in gen_mats]
        if plot_real:
            real_mats = [m[np.ix_(order, order)] for m in real_mats]

        # After reordering, each network occupies a contiguous block
        # with the same size as before. Update idx_lists accordingly so
        # drawing of outlines uses contiguous ranges.
        new_idx_lists = []
        start = 0
        for idxs in per_net_sorted:
            size = len(idxs)
            new_idx_lists.append(np.arange(start, start + size))
            start += size
        idx_lists = tuple(new_idx_lists)

    # Optionally average within/between networks
    if avg_networks:
        n_networks = len(names)

        def avg_fc(mat: np.ndarray) -> np.ndarray:
            out = np.zeros((n_networks, n_networks), dtype=mat.dtype)
            for i, idx_i in enumerate(idx_lists):
                for j, idx_j in enumerate(idx_lists):
                    out[i, j] = mat[np.ix_(idx_i, idx_j)].mean()
            return out

        gen_mats = [avg_fc(m) for m in gen_mats]
        real_mats = [avg_fc(m) for m in real_mats] if plot_real else None

    # Determine color scale across all shown matrices
    all_vals = []
    for m in gen_mats:
        all_vals.append(m)
    if plot_real:
        for m in real_mats:
            all_vals.append(m)
    vmax = np.nanmax([np.nanmax(m) for m in all_vals]) if all_vals else 1.0
    vmin = 0.0

    # Set up figure grid
    n_rows = 2 if plot_real else 1
    n_cols = len(band_items)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5.25 if plot_real else 2.625),
                             gridspec_kw={"wspace": 0.0, "hspace": 0.0}, squeeze=False)

    # Plot helper
    def _draw_network_outlines(ax, idx_lists_local):
        """Draw black square outlines over diagonal blocks for each network.
        Assumes channels are ordered by network and each network occupies a
        contiguous index range between min(idx) and max(idx).
        """
        for idxs in idx_lists_local:
            start = int(np.min(idxs))
            end = int(np.max(idxs))
            size = end - start + 1
            # Rectangle covering the diagonal block for this network
            rect = Rectangle((start - 0.5, start - 0.5),
                             width=size, height=size,
                             fill=False, edgecolor='black', linewidth=0.6)
            ax.add_patch(rect)

    def _plot_row(row_idx: int, mats: list, row_title: str):
        for col_idx, (band_name, _) in enumerate(band_items):
            ax = axes[row_idx, col_idx]
            mat = mats[col_idx]
            im = ax.imshow(mat, cmap="Reds", vmin=vmin, vmax=vmax, interpolation="None")
            # Draw network outlines only when plotting channel-level FC
            if not avg_networks:
                _draw_network_outlines(ax, idx_lists)
            # Ensure pixel-aligned limits for proper outline alignment
            ax.set_xlim(-0.5, mat.shape[1] - 0.5)
            ax.set_ylim(mat.shape[0] - 0.5, -0.5) if ax.images[0].origin == 'upper' else ax.set_ylim(-0.5, mat.shape[0] - 0.5)
            # Column titles on top row
            #if row_idx == 0:
            #    ax.set_title(band_name, fontsize=11)
            # Leftmost y label
            #if col_idx == 0:
            #    ax.set_ylabel(row_title)
            # Ticks/labels
            ax.set_xticks([])
            ax.set_yticks([])
        
    if plot_real:
        _plot_row(0, real_mats, "Real")
        _plot_row(1, gen_mats, "Reconstructed")
    else:
        _plot_row(0, gen_mats, "Reconstructed")
    
    if plot_labels and avg_networks:
        ax = axes[1, 0]
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=90)


    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(os.path.join(fig_path, f"{dataset}/{"reconstruction" if plot_real else "generated"}_batch_fc_s{seed}.svg"), dpi=1000)


def plot_tfr_comparison(
    real_data,
    rec_data,
    dataset,
    anatomy_path=None,
    sfreq=FS_DEFAULT,
    freqs=None,
    n_cycles=None,
    use_coi=False,
    plot_labels=True,
):
    """
    Plot a grid of TFR comparisons across networks (and Whole).

    Layout: 4x8 subplots. Columns are networks (up to 16 incl. Whole).
    Rows are grouped by network halves: rows 0–1 show Real vs. Reconstructed
    for the first 8 networks; rows 2–3 show Real vs. Reconstructed for the
    next 8 networks. Each panel shows channel-averaged power for the first
    subject/window, matching plot_psd selection behavior.
    """

    # frequencies and cycles defaults
    if freqs is None:
        freqs = np.arange(1, 51)
    if n_cycles is None:
        n_cycles = np.maximum(3, freqs / 2.0)

    # Use first window as in plot_psd
    plot_real = isinstance(real_data, np.ndarray)
    real_win = real_data[:, 0] if plot_real else None
    rec_win = rec_data[:, 0]

    # Determine networks and append Whole brain
    network_indices = get_network_indices(dataset, anatomy_path)
    network_indices = dict(network_indices)  # ensure mutable copy
    all_idxs = np.arange(rec_win.shape[1])
    # Compose list and ensure 'Whole' is kept if we need to trim
    items_net = list(network_indices.items())
    whole_item = ("Whole", all_idxs)
    combined = items_net + [whole_item]

    max_panels = 16
    if len(combined) > max_panels:
        # keep first 15 networks + Whole to make 16
        items = combined[:max_panels - 1] + [whole_item]
    else:
        items = combined

    # Precompute TFRs per network for Real and Reconstructed
    real_tfrs = []
    rec_tfrs = []
    times = None

    for _, idxs in items:
        # select first subject and channels of the network
        real_sel = real_win[0, idxs, :] if real_win is not None else None
        rec_sel = rec_win[0, idxs, :]

        # shapes for mne: (n_epochs=1, n_channels, n_times)
        if real_sel is not None:
            real_ep = real_sel[np.newaxis, :, :]
        rec_ep = rec_sel[np.newaxis, :, :]

        # compute power then average across channels
        if real_sel is not None:
            tfr_real = tfr_array_morlet(
                real_ep, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
                output='power', zero_mean=True
            )[0].mean(axis=0)
        else:
            tfr_real = None

        tfr_rec = tfr_array_morlet(
            rec_ep, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', zero_mean=True
        )[0].mean(axis=0)

        # time axis
        if times is None:
            times = np.arange(rec_sel.shape[-1]) / sfreq

        # Optionally mask cone of influence
        if use_coi:
            ncy = np.asarray(n_cycles, float)
            margins = (ncy / (2.0 * freqs)).astype(float)
            t_grid = (np.arange(rec_sel.shape[-1]) / sfreq)[None, :]
            m_grid = margins[:, None]
            coi_mask = (t_grid < m_grid) | ((times[-1] - t_grid) < m_grid)
            if tfr_real is not None:
                tfr_real = np.where(coi_mask, np.nan, tfr_real)
            tfr_rec = np.where(coi_mask, np.nan, tfr_rec)

        real_tfrs.append(tfr_real)
        rec_tfrs.append(tfr_rec)

    # Global color scale across all real and reconstructed panels
    all_imgs = []
    if plot_real:
        for t in real_tfrs:
            if t is not None:
                all_imgs.append(t)
    for t in rec_tfrs:
        all_imgs.append(t)
    vmin = np.nanmin([np.nanmin(t) for t in all_imgs]) if all_imgs else 0.0
    vmax = np.nanmax([np.nanmax(t) for t in all_imgs]) if all_imgs else 1.0
    print(vmin, vmax)

    n_cols = 8
    if plot_real:
        # 4x8 grid
        fig, axes = plt.subplots(4, n_cols, figsize=(20, 5.25),
                                 gridspec_kw={"wspace": 0, "hspace": 0})

        for i, (net_name, _) in enumerate(items):
            row_grp = 0 if i < n_cols else 2
            col = i if i < n_cols else i - n_cols

            ax_real = axes[row_grp, col]
            ax_rec  = axes[row_grp + 1, col]

            if real_tfrs[i] is not None:
                ax_real.imshow(
                    real_tfrs[i], aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='jet', vmin=vmin, vmax=vmax,
                )
            ax_rec.imshow(
                rec_tfrs[i], aspect='auto', origin='lower',
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                cmap='jet', vmin=vmin, vmax=vmax,
            )

            ax_real.text(0.5, 0.9, f"{net_name}", fontsize=12,
                         ha='center', va='center', transform=ax_real.transAxes, color="white")
            
            # put top and bot spines white
            ax_real.spines['top'].set_color('white')
            ax_real.spines['top'].set_linewidth(2)
            ax_rec.spines['bottom'].set_color('white')
            ax_rec.spines['bottom'].set_linewidth(2)

        for r in range(4):
            for c in range(n_cols):
                ax = axes[r, c]
                ax.set_xticks([])
                ax.set_yticks([])

        if plot_labels:
            axes[3, 0].set_xlabel("Time (s)")
            axes[3, 0].set_ylabel("Frequency (Hz)")

    else:
        # 2x8 grid (Recon only)
        fig, axes = plt.subplots(2, n_cols, figsize=(20, 2.625),
                                 gridspec_kw={"wspace": 0, "hspace": 0})

        for i, (net_name, _) in enumerate(items):
            row = 0 if i < n_cols else 1
            col = i if i < n_cols else i - n_cols
            ax = axes[row, col]

            ax.imshow(
                rec_tfrs[i], aspect='auto', origin='lower',
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                cmap='jet', vmin=vmin, vmax=vmax,
            )
            ax.text(0.5, 0.9, net_name, fontsize=14,
                    ha='center', va='center', transform=ax.transAxes, color="white")

            ax.set_xticks([])
            ax.set_yticks([])

            # set top bot spines white
            ax.spines['top'].set_color('white')
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_color('white')
            ax.spines['bottom'].set_linewidth(2)

        if plot_labels:
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Frequency (Hz)")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save
    out_dir = os.path.join(fig_path, dataset)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"reconstruction_batch_tfr_s{seed}.svg" if plot_real else f"generated_batch_tfr_s{seed}.svg"
    plt.savefig(os.path.join(out_dir, fname))


if __name__ == "__main__":
    main()
