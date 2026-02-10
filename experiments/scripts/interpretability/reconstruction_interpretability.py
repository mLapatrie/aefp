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

from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import shutil

from aefp.datasets.meg_dataset import MEGDataset
from aefp.utils.meg_utils import get_network_indices, compute_psd_torch, compute_aec_torch
from aefp.utils.utils import load_autoencoder
from aefp.utils.fingerprinting_utils import get_valid_test_subjects
from aefp.utils.plotting_utils import style_axes


FS_DEFAULT = 150

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
fig_path = os.path.expanduser("~/dev/python/aefp/experiments/.figures/interpretability/reconstruction/")
os.makedirs(fig_path, exist_ok=True)



def main():

    seed = 52
    set_seed(seed)

    dataset = "camcan"
    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"
    data_path = f"/export01/data/{dataset}/"
    out_path = os.path.expanduser(f"~/dev/python/aefp/experiments/.figures/interpretability/{dataset}/reconstruction/")
    os.makedirs(out_path, exist_ok=True)
    anatomy_path = None
    n_subjects = 1
    n_channels_plot = 18
    fs = 150
    fs_cutoff = 50
    network_index = -1
    avg_networks = True
    n_windows = 1 # 60s

    # Choose the color for generated/reconstructed plots in subfunctions
    plot_color = "C0"  # e.g., "red", "blue", "green"

    network_indices = get_network_indices(dataset, None)
    print(network_indices)

    real, gen = load_real_and_recon(
        model_path=model_path,
        data_path=data_path,
        n_subjects=n_subjects,
        block_shape=(200, 900),
        n_windows=n_windows,
    )

    #n_channels_plot = 12
    #plot_traces(real, gen, sample_idx=0, n_channels=n_channels_plot, offset=3.0, fs=fs, gen_color=plot_color) # SEED 1

    network_index = -1
    plot_psd(real, gen, dataset=dataset, anatomy_path=anatomy_path,
             fs=fs, fs_cutoff=fs_cutoff, network_index=network_index,
             gen_color=plot_color) # SEED 52

    #plot_fc_graphs(real, gen, dataset=dataset, anatomy_path=anatomy_path, avg_networks=avg_networks) # SEED 1 & 16
    
    #plot_alpha_peak_topography(real, gen, out_path=out_path, fs=fs) # SEED 4
    
    #plot_tfr_comparison(real, gen, sfreq=fs, freqs=np.arange(1, 31), n_cycles=3, channel_idx=network_indices["SomMotB"][:10], use_coi=False)




def load_real_and_recon(model_path, data_path, n_subjects=1,
                            block_shape=(200, 900), n_windows: int = 1):
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
        for i in range(n_subjects)
    ])

    with torch.no_grad():
        tensor = torch.tensor(real.reshape(-1, real.shape[-2], real.shape[-1]),
                              dtype=torch.float32, device=device)
        print("latent space:", model.encode(tensor.unsqueeze(1)).sample().shape)
        dec = model.decode(model.encode(tensor.unsqueeze(1)).mode())

    return real, dec


def plot_traces(real_data, gen_data,
                sample_idx: int = 0,
                n_channels: int = 20,
                offset: float = 3.0,
                fs: int = FS_DEFAULT,
                real_color: str = "black",
                gen_color: str = "red"):
    """
    Plot raw traces for a real vs. generated example.

    Parameters
    ----------
    real_data, gen_data : np.ndarray
        Arrays of shape (batch, channels, timepoints) or
        (batch, windows, channels, timepoints).  If windows are provided,
        the first window is used.
    sample_idx : int
        Which batch element to plot.
    n_channels : int
        How many channels / timeseries to show.
    offset : float
        Vertical spacing between successive traces.
    fs : int
        Sampling frequency in Hz.
    """

    # If data contain multiple windows, take the first one
    if real_data.ndim == 4:
        real_data = real_data[:, 0]
        gen_data = gen_data[:, 0]

    # Prepare figure + axes for separate real/reconstruction views
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    # Clip to available channels
    n_ch = min(n_channels, real_data.shape[1])

    t = np.arange(real_data.shape[-1]) / fs

    real_data = real_data[:, -n_channels:]
    gen_data = gen_data[:, -n_channels:]
    print(real_data.shape)

    # Plot real / generated with offset
    for i in range(n_ch):
        axes[0].plot(t, real_data[sample_idx, i] + i*offset,
                     color=real_color, alpha=0.8, linewidth=0.75)
        axes[1].plot(t, gen_data[sample_idx, i] + i*offset,
                     color=gen_color,   alpha=0.8, linewidth=0.75)
    
    # Titles & labels
    axes[0].set_title("Original", fontsize=11)
    axes[1].set_title("Decoded", fontsize=11)
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_yticks([])  # hide y‐ticks if you like

    axes[0].set_xlim(0, t[-1])
    axes[1].set_xlim(0, t[-1])

    # remove spines
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "reconstruction_traces.svg"))
    plt.close(fig)

    # Also save an overlapping figure (real vs reconstruction on same axes)
    fig_ov, ax_ov = plt.subplots(1, 1, figsize=(7, 3.0), sharex=True)

    for i in range(n_ch):
        # plot both on same axis with vertical offset per channel
        ax_ov.plot(t, real_data[sample_idx, i] + i*offset,
                   color=real_color, alpha=0.8, linewidth=0.75,
                   label="Original" if i == 0 else None)
        ax_ov.plot(t, gen_data[sample_idx, i] + i*offset,
                   color=gen_color, alpha=0.8, linewidth=0.75,
                   label="Decoded" if i == 0 else None)

    ax_ov.set_title("Real vs Reconstruction (overlap)", fontsize=11)
    ax_ov.set_xlabel("Time (s)")
    ax_ov.set_yticks([])
    ax_ov.set_xlim(0, t[-1])
    # remove spines
    ax_ov.spines['right'].set_visible(False)
    ax_ov.spines['top'].set_visible(False)
    ax_ov.spines['left'].set_visible(False)
    # lightweight legend with single entry for both types
    ax_ov.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "reconstruction_traces_overlap.svg"))
    plt.close(fig_ov)



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


def bayes_factor01_bic(t_stat, df, n_total):
    # BIC approximation to the Bayes factor for the point null.
    return np.sqrt(n_total) * (1.0 + (t_stat**2) / df) ** (-0.5 * n_total)


def plot_psd(real_data, gen_data, dataset, anatomy_path=None,
             fs=FS_DEFAULT, fs_cutoff=50, network_index=0,
             real_color: str = "black", gen_color: str = "red"):
    """
    Plot PSD for selected network(s) comparing real vs. generated data.

    Parameters
    ----------
    real_data, gen_data : np.ndarray
        Arrays of shape (batch, channels, timepoints) or
        (batch, windows, channels, timepoints). If multiple windows are
        provided, the first window is used.
    network_index : int | list[int]
        Index or indices into ``get_network_indices().keys()`` specifying
        which network(s) to plot.
    """
    if real_data.ndim == 4:
        real_data = real_data[:, 0]
        gen_data = gen_data[:, 0]
    # 1) get mapping name → channel-indices
    network_indices = get_network_indices(dataset, anatomy_path)
    items = list(network_indices.items())  # [(name1, idxs1), (name2, idxs2), …]

    # 2) select one or more networks
    if isinstance(network_index, int):
        selected = [items[network_index]]
    elif isinstance(network_index, (list, tuple)):
        selected = [items[i] for i in network_index]
    else:
        raise TypeError("network_index must be int or list of ints")

    # 3) make 2-row figure
    fig, ax = plt.subplots(
        1, 1,
        figsize=(6, 4.5),
    )
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4) loop over each selected network
    for net_name, idxs in selected:
        # subset your data to just that network’s channels
        real_sub = real_data[0, idxs, :]   # shape (n_ch_net, n_time)
        gen_sub  = gen_data[0,  idxs, :]

        # compute PSD + std
        freqs, psd_r, std_r = _psd_avg(real_sub, fs, fs_cutoff=fs_cutoff)
        _,     psd_g, std_g = _psd_avg(gen_sub,  fs, fs_cutoff=fs_cutoff)

        # plot real (solid) + generated (dashed)
        ax.plot(freqs, psd_r,       label=f"{net_name} Real", color=real_color)
        ax.fill_between(freqs, psd_r-std_r, psd_r+std_r,
                             alpha=0.2, label="_nolegend_", color=real_color)
        ax.plot(freqs, psd_g, linestyle="-", label=f"{net_name} Gen", color=gen_color)
        ax.fill_between(freqs, psd_g-std_g, psd_g+std_g,
                             alpha=0.2, label="_nolegend_", color=gen_color)

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

        # mark frequencies with evidence for the null (Jeffreys' scale)
        print(bf01)
        support_null = bf01 >= 3.0
        ax.plot(freqs[support_null], np.ones(support_null.sum()) * (ax.get_ylim()[0] + 0.01),
                marker="*", linestyle="None", color="black", markersize=3,)

    # 5) finalize axes
    ax.set_ylabel("PSD (AU²/Hz)")
    #ax.legend(loc="upper right")

    # plot alpha band range
    ax.axvline(8, color="gray", linestyle="--", linewidth=1)
    ax.axvline(12, color="gray", linestyle="--", linewidth=1)

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "reconstruction_psd.svg"))


def plot_fc_graphs(real_data, gen_data, dataset, anatomy_path=None, avg_networks=False):
    """Plot functional connectivity matrices for real and generated data."""

    network_indices = get_network_indices(dataset, anatomy_path)
    names, idx_lists = zip(*network_indices.items())

    # If data contain multiple subjects/windows, take the first of each
    if real_data.ndim == 4:
        real_data = real_data[0, 0]
        gen_data = gen_data[0, 0]
    elif real_data.ndim == 3:
        real_data = real_data[0]
        gen_data = gen_data[0]

    real_aec = compute_aec_torch(torch.tensor(real_data).unsqueeze(0)).squeeze(0).numpy()
    gen_aec  = compute_aec_torch(torch.tensor(gen_data).unsqueeze(0)).squeeze(0).numpy()

    if avg_networks:
        # Average FC within/between networks
        n_networks = len(names)

        def avg_fc(mat: np.ndarray) -> np.ndarray:
            out = np.zeros((n_networks, n_networks))
            for i, idx_i in enumerate(idx_lists):
                for j, idx_j in enumerate(idx_lists):
                    out[i, j] = mat[np.ix_(idx_i, idx_j)].mean()
            return out

        real_aec = avg_fc(real_aec)
        gen_aec = avg_fc(gen_aec)

        vmin = 0
        vmax = 0.4

        fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.625), gridspec_kw={'wspace': 0.05})
        for ax, mat, title in zip(axes, (real_aec, gen_aec), ("Original", "Decoded")):
            im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=11)
            ax.set_xticks(np.arange(n_networks))
            ax.set_xticklabels(names, rotation=90)
            #ax.set_xticks([])

        axes[0].set_yticks(np.arange(n_networks))
        axes[0].set_yticklabels(names)
        axes[1].set_yticks([])  # hide y‐ticks for second plot


        #cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        #cbar.set_label("AEC correlation")

        print(np.max(real_aec), np.min(real_aec), np.mean(real_aec))
        print(np.max(gen_aec), np.min(gen_aec), np.mean(gen_aec))
        #plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(fig_path, "reconstruction_aec.svg"))
        return

    # --- Parcel-level plotting as before ---
    order = np.concatenate(idx_lists)
    counts = [len(idxs) for idxs in idx_lists]
    centers = [sum(counts[:i]) + cnt / 2 for i, cnt in enumerate(counts)]

    real_aec = real_aec[order][:, order]
    gen_aec = gen_aec[order][:, order]

    absmax = max(np.abs(real_aec).max(), np.abs(gen_aec).max())
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.05})
    for i, (ax, mat, title) in enumerate(zip(axes, (real_aec, gen_aec), ("Real FC", "Generated FC"))):
        im = ax.imshow(mat, cmap="RdBu", norm=norm, interpolation="nearest")
        ax.set_title(title)

        start = 0
        for size in counts:
            rect = Rectangle(
                (start - 0.5, start - 0.5),
                width=size,
                height=size,
                fill=False,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)
            start += size

        ax.set_xticks(centers)
        ax.set_xticklabels(names, rotation=90)
        if i == 0:
            ax.set_yticks(centers)
            ax.set_yticklabels(names)
        else:
            ax.set_yticks([])

    #cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    #cbar.set_label("AEC correlation")
    print(np.max(real_aec), np.min(real_aec), np.mean(real_aec))
    print(np.max(gen_aec), np.min(gen_aec), np.mean(gen_aec))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "reconstruction_aec.svg"))


def plot_alpha_peak_topography(real, generated, out_path, fs=FS_DEFAULT, alpha_band=(8, 13)):
    """Plot peak alpha frequency for each parcel on the brain.

    For each of ``real`` and ``generated`` data, the PSD is computed for
    multiple consecutive windows of a single subject, averaged, and the peak
    alpha frequency is projected onto the brain surface using an external R
    script.

    Parameters
    ----------
    real, generated : np.ndarray
        Arrays of shape (subjects, windows, channels, timepoints) or
        (windows, channels, timepoints).
    out_path : str
        Directory where the resulting figures should be stored.
    fs : int
        Sampling frequency.
    alpha_band : tuple
        Frequency range for the alpha band in Hz.
    """

    print(real.shape, generated.shape)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    labels_path = os.path.join(repo_root, "Schaefer2018_200_17networks_labels.txt")
    r_script = os.path.join(repo_root, "plot_schaefer.R")

    for data, label in zip((real, generated), ("real", "generated")):
        # Select first subject if necessary
        if data.ndim == 4:
            data = data[0]

        # Compute PSD for each window and average
        psds = []
        for win in data:
            psd, freqs = compute_psd_torch(torch.tensor(win).unsqueeze(0), fs=fs, log=False)
            psds.append(psd.squeeze(0).numpy())
        psd_mean = np.mean(psds, axis=0)
        freqs = freqs.numpy()

        mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_psd = psd_mean[:, mask]
        alpha_freqs = freqs[mask]
        peak_idx = np.argmax(alpha_psd, axis=1)
        peak_freqs = alpha_freqs[peak_idx]
        peak_freqs[:] = 0.0
        print(peak_freqs)

        tmp_dir = tempfile.mkdtemp()
        try:
            pd.DataFrame(peak_freqs).to_csv(os.path.join(tmp_dir, f"alpha_peak_{label}.csv"), index=False)
            subprocess.run(
                ["Rscript", r_script, labels_path, tmp_dir, out_path, "RdBu"],
                check=True,
            )
        except Exception as e:
            print(f"Failed to run R script: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def plot_tfr_comparison(
    real_data, gen_data, sfreq, freqs=None, n_cycles=None,
    channel_idx=0, baseline=(None, 0.0), use_coi=True
):

    if freqs is None:
        freqs = np.arange(1, 51)  # 1–50 Hz

    if n_cycles is None:
        # smaller at low f to avoid extreme edge bleed on short epochs
        n_cycles = np.maximum(3, freqs / 2.0)
    
    if type(channel_idx) is not list:
        channel_idx = [channel_idx]

    # select one epoch+channel -> shapes (1,1,T)
    real = real_data[0, :1, channel_idx, :] # take 30s from sub 0
    #real = np.concatenate(real, axis=-1)[np.newaxis, :]  # (1, 1, 5*T)
    gen  = gen_data [0, :1, channel_idx, :] # take 30s from sub 0
    #gen = np.concatenate(gen, axis=-1)[np.newaxis, :]  # (1, 1, 5*T)

    # compute power
    tfr_real = tfr_array_morlet(real, sfreq=sfreq, freqs=freqs,
                                n_cycles=n_cycles, output='power',
                                zero_mean=True)[0].mean(axis=0)
    tfr_gen  = tfr_array_morlet(gen,  sfreq=sfreq, freqs=freqs,
                                n_cycles=n_cycles, output='power',
                                zero_mean=True)[0].mean(axis=0)

    # baseline to dB (relative change)
    times = np.arange(real.shape[-1]) / sfreq
    tfr_real_db = tfr_real #rescale(tfr_real, times, baseline=baseline, mode='logratio') * 10
    tfr_gen_db  = tfr_gen #rescale(tfr_gen,  times, baseline=baseline, mode='logratio') * 10

    # optional cone of influence mask
    if use_coi:
        # margin(f) ≈ n_cycles/(2f) seconds; handle scalar or per-freq n_cycles
        ncy = np.asarray(n_cycles, float)
        if ncy.ndim == 0:
            margins = (ncy / (2.0 * freqs)).astype(float)
        else:
            margins = (ncy / (2.0 * freqs)).astype(float)
        # build mask: True = should be masked
        T = times[-1] - times[0]
        t_grid = times[None, :]
        m_grid = margins[:, None]
        coi_mask = (t_grid - times[0] < m_grid) | (times[-1] - t_grid < m_grid)

        # mask with NaNs so imshow leaves it blank
        tfr_real_db = np.where(coi_mask, np.nan, tfr_real_db)
        tfr_gen_db  = np.where(coi_mask,  np.nan, tfr_gen_db)

    # shared scale for fair comparison (ignore NaNs)
    vmin = np.nanmin([tfr_real_db, tfr_gen_db])
    vmax = np.nanmax([tfr_real_db, tfr_gen_db])

    fig, axes = plt.subplots(2, 1, figsize=(5.25, 5.8), sharex=True,
                             gridspec_kw={'height_ratios':[1,1]})
    for ax, img, title in zip(axes, (tfr_real_db, tfr_gen_db), ("Original", "Decoded")):
        im = ax.imshow(img, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       cmap='jet')#, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Frequency (Hz)")
        #fig.colorbar(im, ax=ax, label='Power (AU)')
    
    print(vmin, vmax)

    axes[-1].set_xlabel("Time (s)")
    
    # remove spines
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig("experiments/.figures/interpretability/reconstruction/tfr_comparison.svg")
    #plt.show()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)



if __name__ == "__main__":
    main()
