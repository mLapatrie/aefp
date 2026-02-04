import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import pandas as pd

from glob import glob
from tqdm import tqdm
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D
from numpy.linalg import lstsq

from aefp.utils.plotting_utils import style_axes
from aefp.utils.utils import load_autoencoder
from aefp.utils.fingerprinting_utils import (
    get_valid_test_subjects,
    fingerprint,
)
from experiments.scripts.interpretability.helpers.interpretability_helpers import (
    get_latent_pca,
    extract_features,
    BANDS_DEF,
    get_network_indices,
    load_subject_embeddings,
    icc_3_1,
    reduce_by_groups_nan,
    yeo_reduced_labels,
    plot_colorbar,
)
from experiments.scripts.fingerprinting.helpers.fingerprinting_helpers import (
    bootstrap_metrics,
    butter_bandstop,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

def get_windows(
    subject,
    dataset_root,
    window_size,
    step_size,
):
    session_paths = sorted(
        glob(os.path.join(dataset_root, subject, "ses-*", "meg", "rest", "source_200.pt"))
    )

    all_windows = []
    for ses_path in session_paths:
        ses_data = torch.load(ses_path, weights_only=False)
        _, T = ses_data.shape

        if T < window_size:
            continue

        # slide window
        starts = range(0, T - window_size + 1, step_size)
        for s in starts:
            end = s + window_size
            block = ses_data[:, s:end]

            # normalize block
            block = (block - block.mean()) / block.std()
            all_windows.append(block)
    
    if len(all_windows) == 0:
        raise RuntimeError(f"No windows found for subject {subject}")
    
    windows = torch.stack(all_windows)
    return windows


def encode_windows(model, windows, pca, scaler, pca_transform=True):
    
    with torch.no_grad():
        z_chunks = []
        for xb in windows.split(BATCH_SIZE, dim=0):
            zb = model.encode(xb.unsqueeze(1).to(device)).mode()
            z_chunks.append(zb.cpu())
        z = torch.cat(z_chunks, dim=0)
    z = z.cpu().numpy().reshape(len(windows), -1) # flatten latents

    if not pca_transform:
        return z

    # project in pc space
    z_norm = scaler.transform(z)
    z_pca = pca.transform(z_norm)

    return z_pca


def decode_latents(model, z_pca, pca, scaler, latent_shape):
    
    # inverse pca
    z_norm = pca.inverse_transform(z_pca)
    z = scaler.inverse_transform(z_norm)
    z = z.reshape(len(z_pca), *latent_shape)

    # decode
    with torch.no_grad():
        x_chunks = []
        for zb in torch.tensor(z, dtype=torch.float32).split(BATCH_SIZE, dim=0):
            xb = model.decode(zb.to(device))
            x_chunks.append(xb.cpu())
        x_hat = torch.cat(x_chunks, dim=0)
    
    return x_hat


def compute_feature_importance(
    model,
    test_subjects,
    pca,
    scaler,
    max_pc,
    window_size,
    step_size,
    dataset_root,
    latent_shape,
    network_indices,
):
    
    importances = []

    total_win = 0
    for subject in tqdm(test_subjects):

        # get windows
        windows = get_windows(subject, dataset_root, window_size, step_size)
        total_win += len(windows)

        # encode windows and project in pc space
        z_pca = encode_windows(model, windows, pca, scaler)

        # extract features (w/o occlusion)
        x_hat = decode_latents(model, z_pca, pca, scaler, latent_shape)
        x_hat = x_hat.squeeze(1)  # (n_windows, n_rois, window_size)

        # occlude pc, reconstruct (w/ w/o occ) and extract features
        base_features = np.stack([extract_features(x_h, network_indices, avg_networks_aec=True, avg_networks_psd=False, fc_bands=True) for x_h in x_hat])
        base_features_std = base_features.std(axis=0)

        pc_importances = []
        for pc in range(max_pc):
            
            # occlude pc from z_pca
            z_pca_occluded = z_pca.copy()
            z_pca_occluded[:, pc] = 0.0

            # decode occluded
            x_hat_occluded = decode_latents(model, z_pca_occluded, pca, scaler, latent_shape)
            x_hat_occluded = x_hat_occluded.squeeze(1)  # (n_windows, n_rois, window_size)

            # extract features (w/ occlusion)
            occluded_features = np.stack([extract_features(x_h, network_indices, avg_networks_aec=True, avg_networks_psd=False, fc_bands=True) for x_h in x_hat_occluded])

            # compute feature importance as absolute difference in z score
            imp_all = np.abs((base_features - occluded_features) / base_features_std)
            imp = imp_all.mean(axis=0)  # average over windows

            pc_importances.append(imp)
        
        pc_importances = np.stack(pc_importances)  # (n_pcs, n_features)
        importances.append(pc_importances)
    print(f"Total windows: {total_win}")
    importances = np.stack(importances)  # (n_subjects, n_pcs, n_features)

    return importances


def build_X(
    model,
    subjects,
    dataset_root,
    window_size, # window lengths
    segment_size, # total fingerprint length
    step_size, # step between windows
    latent_shape,
    occlude_indices=None, # pc to occlude
    saved_windows_beg=None,
    saved_windows_end=None,
):
    
    X = []
    all_windows_beg = []
    all_windows_end = []
    for sub_idx, subject in enumerate(subjects):
        if saved_windows_beg is None or saved_windows_end is None:
            windows = get_windows(subject, dataset_root, window_size, step_size)
            
            N_windows_per_fp = (segment_size - window_size) // step_size + 1
            windows_beg = windows[:N_windows_per_fp]
            windows_end = windows[-N_windows_per_fp:]
            all_windows_beg.append(windows_beg)
            all_windows_end.append(windows_end)
        else:
            windows_beg = saved_windows_beg[sub_idx]
            windows_end = saved_windows_end[sub_idx]

        # encode windows
        z_pca_beg = encode_windows(model, windows_beg, pca, scaler)
        z_pca_end = encode_windows(model, windows_end, pca, scaler)

        z_pca_beg[:, 50:] = 0.0
        z_pca_end[:, 50:] = 0.0
 
        if occlude_indices is not None:
            z_pca_beg[:, occlude_indices] = 0.0
            z_pca_end[:, occlude_indices] = 0.0
        
        # unpca
        z_beg = pca.inverse_transform(z_pca_beg)
        z_end = pca.inverse_transform(z_pca_end)
        z_beg = scaler.inverse_transform(z_beg)
        z_end = scaler.inverse_transform(z_end)

        #
        ## decode windows
        #x_hat_beg = decode_latents(model, z_pca_beg, pca, scaler, latent_shape)
        #x_hat_end = decode_latents(model, z_pca_end, pca, scaler, latent_shape)
        #x_hat_beg = x_hat_beg.squeeze(1)  # (n_windows, n_rois, window_size)
        #x_hat_end = x_hat_end.squeeze(1)  # (n_windows, n_rois, window_size
        #
        ## reencode
        #z_beg = encode_windows(model, x_hat_beg, pca, scaler, pca_transform=False)
        #z_end = encode_windows(model, x_hat_end, pca, scaler, pca_transform=False)

        # average to create fingerprint
        z_fp_beg = z_beg.mean(axis=0)
        z_fp_end = z_end.mean(axis=0)

        X.append((z_fp_beg, z_fp_end))
    
    X = np.array(X)  # (n_subjects, 2, n_features)
    return X, (all_windows_beg, all_windows_end)


def test_reconstruction_fingerprinting(
    model,
    subjects,
    dataset_root,
    window_size,
    segment_size,
    step_size,
    latent_shape,
    n_pcs,
):
    X, (w_b, w_e) = build_X(
        model,
        subjects,
        dataset_root,
        window_size,
        segment_size,
        step_size,
        latent_shape,
        occlude_indices=None, # no occlusion
    )
    base_diff_mean, base_diff_std = bootstrap_metrics(X, metric_idx=3, bootstraps=100)
    print(f"Base differentiability {base_diff_mean:.4f} +/- {base_diff_std:.4f}")
    
    diffs = []
    for pc in range(n_pcs):
        X, _ = build_X(
            model,
            subjects,
            dataset_root,
            window_size,
            segment_size,
            step_size,
            latent_shape,
            occlude_indices=[pc], # occlude pc
            saved_windows_beg=w_b,
            saved_windows_end=w_e,
        )
        diff_mean, diff_std = bootstrap_metrics(X, metric_idx=3, bootstraps=100)
        print(f"PC {pc}: differentiability {diff_mean:.4f} +/- {diff_std:.4f}")
        diffs.append((diff_mean, diff_std))
    
    diffs_drops = np.array([(base_diff_mean - d[0], np.sqrt(base_diff_std**2 + d[1]**2)) for d in diffs]) # compute difference of two normal distributions
    return diffs_drops # (n_pcs, 2)


def test_reconstruction(
    model,
    subjects,
    dataset_root,
    window_size,
    segment_size,
    step_size,
    latent_shape,
    n_pcs,
):
    
    # encode all subject windows
    # for each pc, occlude and reconstruct
    # reencode and compute latent space error for the occluded pc

    zocc_zhatocc = []
    zocc_zhat = []
    z_zhatocc = []
    zrand_zhatocc = []
    zocc_z = []
    for subject in tqdm(subjects):

        windows = get_windows(subject, dataset_root, window_size, step_size)

        z_pca = encode_windows(model, windows, pca, scaler)
        
        zocc_zhatocc_pc = []
        zocc_zhat_pc = []
        z_zhatocc_pc = []
        zrand_zhatocc_pc = []
        zocc_z_pc = []
        for pc in range(0, n_pcs+1):
            pc_sd = pca.explained_variance_[pc] ** 0.5

            z_pca_occluded = z_pca.copy()
            z_pca_occluded[:, pc] = 0.0

            # decode normal
            x_hat = decode_latents(model, z_pca, pca, scaler, latent_shape)
            x_hat = x_hat.squeeze(1)  # (n_windows, n_rois, window_size)

            # decode occluded
            x_hat_occluded = decode_latents(model, z_pca_occluded, pca, scaler, latent_shape)
            x_hat_occluded = x_hat_occluded.squeeze(1)  # (n_windows, n_rois, window_size)

            # reencode
            z_pca_reencoded = encode_windows(model, x_hat, pca, scaler)
            z_occ_pca_reencoded = encode_windows(model, x_hat_occluded, pca, scaler)

            # calculate error
            zocc_zhatocc_err = np.abs(z_occ_pca_reencoded[:, pc]) # 0 - zhatocc
            zocc_zhat_err = np.abs(z_pca_reencoded[:, pc]) # 0 - zhat
            z_zhatocc_err = np.abs(z_pca[:, pc] - z_occ_pca_reencoded[:, pc])
            zrand_zhatocc_err = np.abs(z_pca[np.random.permutation(len(z_pca)), pc] - z_occ_pca_reencoded[:, pc]) # random latent vector from same sub - zhatocc
            zocc_z_err = np.abs(z_pca[:, pc]) # 0 - z

            # scale errors by pc sd
            zocc_zhatocc_err /= pc_sd
            zocc_zhat_err /= pc_sd
            z_zhatocc_err /= pc_sd
            zrand_zhatocc_err /= pc_sd
            zocc_z_err /= pc_sd

            zocc_zhatocc_pc.append(zocc_zhatocc_err)
            zocc_zhat_pc.append(zocc_zhat_err)
            z_zhatocc_pc.append(z_zhatocc_err)
            zrand_zhatocc_pc.append(zrand_zhatocc_err)
            zocc_z_pc.append(zocc_z_err)
        
        zocc_zhatocc.append(zocc_zhatocc_pc)
        zocc_zhat.append(zocc_zhat_pc)
        z_zhatocc.append(z_zhatocc_pc)
        zrand_zhatocc.append(zrand_zhatocc_pc)
        zocc_z.append(zocc_z_pc)

    zocc_zhatocc = np.concatenate(zocc_zhatocc, axis=1)
    zocc_zhat = np.concatenate(zocc_zhat, axis=1)
    z_zhatocc = np.concatenate(z_zhatocc, axis=1)
    zrand_zhatocc = np.concatenate(zrand_zhatocc, axis=1)
    zocc_z = np.concatenate(zocc_z, axis=1)
        
    return zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z


def plot_latent_errors(zocc_zhatocc, z_zhatocc, zrand_zhatocc, fig_path):

    def _clean(a):
        a = np.asarray(a).ravel()
        return a[np.isfinite(a)]
    def _ecdf(a):
        x = np.sort(a)
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    a0 = _clean(zocc_zhatocc)   # occ err to 0
    #a1 = _clean(zocc_zhat)      # err to 0
    a2 = _clean(z_zhatocc)      # err to original
    a3 = _clean(zrand_zhatocc)  # err to random
    #a4 = _clean(zocc_z)         # occ err to z

    x0, y0 = _ecdf(a0)
    #x1, y1 = _ecdf(a1)
    x2, y2 = _ecdf(a2)
    x3, y3 = _ecdf(a3)
    #x4, y4 = _ecdf(a4)
    #m0, m1, m2, m3, m4 = np.median(a0), np.median(a1), np.median(a2), np.median(a3), np.median(a4)
    m0, m2, m3 = np.median(a0), np.median(a2), np.median(a3)


    colors = ["crimson", "gray", "gray", "gray"]
    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    l0, = ax.plot(x0, y0, label="Error relative to zeroed target", color=colors[0], linewidth=2)
    #l1, = ax.plot(x1, y1, label="Error to 0", color=colors[1], linewidth=2)
    l2, = ax.plot(x2, y2, label="Error relative to original latent", color=colors[2], linewidth=2)
    l3, = ax.plot(x3, y3, label="Error relative to same-participant latent", color=colors[3], linewidth=2)
    #l4, = ax.plot(x4, y4, label="Z Error to 0", color=colors[4], linewidth=2)
    for i, m in enumerate((m0, m2, m3)):
        ax.axvline(m, linestyle="--", linewidth=2, c=colors[i])

    # add proxy line for legend
    proxy_line = Line2D([0], [0], color="k", linestyle="--", linewidth=2, label="Median error")

    ax.set_xlabel("Absolute error")
    ax.set_ylabel("ECDF")
    #ax.set_title("ECDF of absolute errors in latent space")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", handles=[l0, l2, l3, proxy_line])
    ax.set(xlim=(0, 2.5), ylim=(0, 1))

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, "latent_reconstruction_errors.svg"))
    plt.close()


def _reconstruct_connectome(flat, n):
    mat = np.zeros((n, n), dtype=flat.dtype)
    idx = np.triu_indices(n, 1)
    mat[idx] = flat
    mat = mat + mat.T
    return mat


def plot_group_importances(importance_array, network_indices, fig_path, reduce_networks=False):
    networks = sorted(network_indices.keys())
    n_networks_imp = len(networks)
    n_rois = sum(len(v) for v in network_indices.values())
    n_bands = len(BANDS_DEF)
    aec_len_band = n_networks_imp * (n_networks_imp - 1) // 2
    aec_len_total = aec_len_band * n_bands

    mean_imps = importance_array.mean(axis=0)
    for pc in range(mean_imps.shape[0]):
        n_networks = n_networks_imp
        imps = mean_imps[pc]
        aec_flat = imps[:aec_len_total]
        psd_flat = imps[aec_len_total:]

        aec_mats = []
        for b in range(n_bands):
            start = b * aec_len_band
            mat = _reconstruct_connectome(
                aec_flat[start : start + aec_len_band], n_networks
            )
            aec_mats.append(mat)

        if reduce_networks:
            aec_mats = [reduce_by_groups_nan(mat, exclude_diag=False, reduction_type="max") for mat in aec_mats]
            n_networks = len(yeo_reduced_labels)
            networks = yeo_reduced_labels

        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        for ax, mat, band in zip(axes.ravel(), aec_mats, BANDS_DEF.keys()):
            lower = np.tril_indices(n_networks, k=-1)
            mat[lower] = np.nan
            im = ax.imshow(mat, cmap="magma")
            ax.set_title(band, fontsize=11)
            ax.set_xticks(range(n_networks))
            ax.set_xticklabels(networks, rotation=45, ha="right", fontsize="x-small")
            ax.set_yticks(range(n_networks))
            ax.set_yticklabels(networks, fontsize="x-small")
            ax.set_xlim(-0.5, n_networks - 0.5)
            ax.set_ylim(n_networks - 0.5, -0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            vmin, vmax = np.nanmin(mat), np.nanmax(mat)
            ticks = np.round(np.linspace(vmin, vmax, 4), 3)
            cbar.set_ticks(ticks)
            fig.canvas.draw()  # ensure layout is computed
            bbox = cbar.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            print(f"Colorbar size: {width:.2f} x {height:.2f} inches")
            # remove spine from colorbar
            im.axes.spines['top'].set_visible(False)
            im.axes.spines['right'].set_visible(False)
            im.axes.spines['left'].set_visible(False)
            im.axes.spines['bottom'].set_visible(False)

        # remove x ticks from top row
        for ax in axes[0, :]:
            ax.set_xticks([])

        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f"pc_{pc + 1}_aec_importance.svg"))
        plt.close(fig)

        psd_mat = psd_flat.reshape(n_rois, n_bands)
        tmp_dir = os.path.join(fig_path, f"pc_{pc+1}", f"tmp_pc_{pc + 1}")
        os.makedirs(tmp_dir, exist_ok=True)

        height = 1.383622642
        width = 0.06588679245
        for b, band in enumerate(BANDS_DEF.keys()):
            psd_band = psd_mat[:, b]
            tmp_csv = os.path.join(tmp_dir, f"psd_{band}.csv")
            pd.DataFrame(psd_band).to_csv(tmp_csv, index=False)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            labels_path = os.path.join(repo_root, "Schaefer2018_200_17networks_labels.txt")
            r_script = os.path.join(repo_root, "plot_schaefer.R")
            try:
                output = subprocess.run(
                    ["Rscript", r_script, labels_path, tmp_dir, os.path.join(fig_path, f"pc_{pc + 1}"), "magma"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                result = output.stdout.strip()
                min = float(result.split("=")[2].split(" ")[0])
                max = float(result.split("=")[3])
                print(result, result.split("="), min, max)
                plot_colorbar(
                    OUT_PATH=os.path.join(fig_path, f"pc_{pc + 1}", f"{band}_colorbar.svg"),
                    WIDTH_IN=width,
                    HEIGHT_IN=height,
                    VMIN=min,
                    VMAX=max,
                    CMAP="magma",
                    ORIENTATION="vertical",
                    TICK_SIDE="right",
                )

            except Exception as e:
                print(f"Failed to run R script: {e}")
            finally:
                if os.path.exists(tmp_csv):
                    os.remove(tmp_csv)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


def plot_pca_icc(emb_dir, pca_path: str, out_dir: str, max_components: int = 500) -> np.ndarray:
    """Plot PCA components' ICC with raw bars, a smoothed line, and an a/k + c fit."""
    # PCA + data
    pca, scaler, _ = get_latent_pca(emb_dir, pca_path, 0.95)
    embeddings, _ = load_subject_embeddings(emb_dir)

    # shape: (n_subjects, 2, feature_dim) using first/last 5 segments as "raters"
    X = np.array([[embeddings[i][:5].mean(axis=0), embeddings[i][-5:].mean(axis=0)]
                  for i in range(len(embeddings))])
    X_flat = np.concatenate(X, axis=0)
    X_std = scaler.transform(X_flat)
    Z_flat = pca.transform(X_std)
    Z = Z_flat.reshape(len(embeddings), 2, -1)

    # ICC per component
    ncomp = min(Z.shape[2], max_components)
    iccs = np.array([icc_3_1(Z[:, :, comp]) for comp in range(ncomp)], dtype=float)

    # helpers
    def rolling_median(y, w):
        w = int(w)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        yp = np.pad(y, (pad, pad), mode='edge')
        out = np.empty_like(y, dtype=float)
        for i in range(len(y)):
            out[i] = np.median(yp[i:i + w])
        return out

    def fit_power_shift(y, alphas=(0.8, 1.0, 1.2), k0_max=40, k0_step=1):
        """
        Fit y ≈ A*(k+k0)^(-alpha) + c by grid over alpha, k0.
        Given alpha,k0 it is linear in A,c -> solve by least squares.
        Returns best yhat and params with highest r^2.
        """
        k = np.arange(1, len(y) + 1, dtype=float)
        best = {"r2": -np.inf}
        for alpha in alphas:
            for k0 in range(0, min(k0_max, len(y)//2) + 1, k0_step):
                basis = (k + k0) ** (-alpha)
                X = np.vstack([basis, np.ones_like(basis)]).T  # columns: basis, 1
                A, c = lstsq(X, y, rcond=None)[0]
                yhat = A * basis + c
                sst = np.sum((y - y.mean())**2)
                sse = np.sum((y - yhat)**2)
                r2 = 1.0 - sse / sst if sst > 0 else 0.0
                if r2 > best["r2"]:
                    best = {"yhat": yhat, "A": A, "alpha": alpha, "k0": k0, "c": c, "r2": r2}
        return best

    # smoothing and fit
    w = max(9, len(iccs) // 20)  # ~5% window
    icc_smooth = rolling_median(iccs, w=w)
    fit = fit_power_shift(iccs, alphas=(0.8, 1.0, 1.2), k0_max=40, k0_step=1)
    yhat, r2 = fit["yhat"], fit["r2"]


    # plot
    split = 'train' if emb_dir.rstrip('/').split('/')[-1] == 'train' else 'test'
    fig, ax = plt.subplots(figsize=(7, 5.25))
    k = np.arange(1, len(iccs) + 1)

    bars = ax.bar(k, iccs, width=1.0, alpha=0.25, color="k", label="ICC by component")
    ln_smooth, = ax.plot(k, icc_smooth, linewidth=2, color="k", label="Smoothed ICC")
    ln_fit,    = ax.plot(k, yhat, linestyle="--", linewidth=2, color="crimson", label=r"Power-law fit $C\,k^{-\alpha}$")

    ax.set(ylim=(0,1), xlim=(0, len(iccs)+1), xlabel="PCA component (k)", ylabel="ICC")
    ax.grid(True, alpha=0.3)

    # clean, ordered legend: bars -> smooth -> fit
    ax.legend(handles=[bars, ln_smooth, ln_fit], loc="upper right", frameon=False, title="Series")

    # keep legend uncluttered; show r² in a corner
    ax.text(0.98, 0.3, rf"$r^2={fit['r2']:.3f}$", transform=ax.transAxes, ha="right", va="bottom")

    # remove top/right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_icc_{split}.svg"), dpi=200)
    plt.close()

    return iccs


def main(
    root_dir,
    model_path,
    out_path,
    fig_path,
    n_subjects,
    window_size,
    step_size,
    max_pc,
    network_indices,
):
    
    # for every pc up to max_pc, compute feature importance over feature set
    # occlude top_k features and bottom_k features and compare distributions of differentiability drop
    # save t stats and p values
    # plot t stats over pcs with stars for significant pcs

    # feature importance is the difference in z score between reconstructed with and without occlusion
    # the z score is computed over the distribution of features in the set of reconstructed windows (without occlusion)

    ###################

    # load model
    model, cfg, sub_dict = load_autoencoder(model_path)
    model.to(device).eval()
    
    test_subjects, _ = get_valid_test_subjects(sub_dict, root_dir, same_session=True)
    test_subjects = test_subjects[:n_subjects]
    print("Using subjects:", test_subjects)
    
    # pass dummy to get latent shape
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 200, window_size, device=device)
        latent_shape = model.encode(dummy).mode().shape[1:]

    ## compute feature importance
    #importances = compute_feature_importance(
    #    model,
    #    test_subjects,
    #    pca,
    #    scaler,
    #    max_pc,
    #    window_size,
    #    step_size,
    #    root_dir,
    #    latent_shape,
    #    network_indices,
    #)  # (n_subjects, n_pcs, n_features)
    #np.save(os.path.join(out_path, f"pc_occlusion_importances_{n_subjects}subs.npy"), importances)
    #
    #plot_group_importances(importances, network_indices, fig_path)
    #
    #assert 1 == 2, "debug stop"

    # test importance of features
    zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z = test_reconstruction(
        model,
        test_subjects,
        root_dir,
        window_size,
        segment_size=4500, # 30s
        step_size=step_size,
        latent_shape=latent_shape,
        n_pcs=max_pc,
    )
    np.save(os.path.join(out_path, f"pc_occlusion_latent_errors_{n_subjects}subs_50.npy"), (zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z))

    plot_latent_errors(zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z, fig_path)


def print_pc_occlusion_summary(
    importance_array: np.ndarray,
    network_indices: dict,
    dataset: str = "camcan",
    pc_base: int = 0,
    avg_networks: bool = False,
    fc_bands: bool = True,
):
    """
    Print a concise summary of feature importance from latent occlusions.

    Expects ``importance_array`` with shape (subjects, PCs, features) where features
    are ordered as [AEC (possibly per-band), then PSD (per-band)]. Computes mean ± SD
    across subjects (the first dimension).
    """
    from scipy.stats import spearmanr

    S, P, F = importance_array.shape
    band_names = list(BANDS_DEF.keys())  # e.g., [broadband, delta, theta, alpha, beta, gamma]
    n_bands = len(band_names)

    # Determine entity count (networks or parcels) from network_indices
    n_parcels = sum(len(v) for v in network_indices.values())
    n_networks = len(network_indices)
    n_entities = n_networks if avg_networks else n_parcels

    # AEC/PSD segment length with light inference if flags mismatch the data
    def compute_lengths(use_fc_bands: bool):
        edges = n_entities * (n_entities - 1) // 2
        a_len = (n_bands * edges) if use_fc_bands else edges
        p_len = n_entities * n_bands
        return edges, a_len, p_len

    aec_edges, aec_len, psd_len = compute_lengths(fc_bands)
    if aec_len + psd_len > F:
        # Try interpreting AEC as broadband-only
        aec_edges, aec_len, psd_len = compute_lengths(False)
        fc_bands = False

    # Helper: split a single importance vector into AEC, PSD parts
    def split_feats(vec):
        aec = vec[:aec_len]
        psd = vec[aec_len : aec_len + psd_len]
        return aec, psd

    # Use PC1 by default (pc_base index)
    pc1 = np.asarray(importance_array[:, pc_base], float)  # (S, F)

    # --- AEC band-wise means (± SD) ---
    aec_band_means = None
    aec_broadband_mean = None
    if fc_bands:
        # shape per subject: (n_bands, aec_edges)
        per_subj_band_means = []
        for s in range(S):
            aec_s, _ = split_feats(pc1[s])
            aec_s = aec_s.reshape(n_bands, aec_edges)
            per_subj_band_means.append(aec_s.mean(axis=1))
        per_subj_band_means = np.stack(per_subj_band_means)  # (S, n_bands)
        aec_band_means = (np.nanmean(per_subj_band_means, axis=0), np.nanstd(per_subj_band_means, axis=0))
        # Individual entries
        bb_idx = band_names.index("broadband") if "broadband" in band_names else 0
        aec_broadband_mean = (float(aec_band_means[0][bb_idx]), float(aec_band_means[1][bb_idx]))

    # --- AEC network-wise ranking (mean across non-gamma bands and edges incident to network) ---
    def aec_network_scores_for_subject(vec):
        aec_s, _ = split_feats(vec)
        if fc_bands:
            # average across non-gamma bands first
            try:
                gamma_idx = band_names.index("gamma")
            except ValueError:
                gamma_idx = None
            use_band_idx = [i for i in range(n_bands) if i != gamma_idx]
            aec_s = aec_s.reshape(n_bands, aec_edges)
            aec_s = aec_s[use_band_idx].mean(axis=0)  # (aec_edges,)
        # Reconstruct full adjacency (entities x entities)
        mat = np.zeros((n_entities, n_entities), float)
        iu = np.triu_indices(n_entities, k=1)
        mat[iu] = aec_s
        mat = mat + mat.T
        # If parcel-level, aggregate to networks
        if not avg_networks:
            nets = sorted(network_indices.keys())
            scores = []
            for net in nets:
                idx = np.asarray(network_indices[net], int)
                # mean of edges incident to the network (exclude self-diagonal)
                block = mat[idx][:, :]
                # include connections from idx to all parcels
                vals = np.concatenate([mat[idx].reshape(-1)])
                # remove self-diagonal elements explicitly
                vals = vals.reshape(len(idx), -1)
                for i in range(len(idx)):
                    vals[i, idx[i]] = np.nan
                vals = vals.reshape(-1)
                scores.append(np.nanmean(vals))
            return np.array(scores)
        else:
            # Network-level entities: use row means excluding diagonal
            scores = []
            for i in range(n_entities):
                row = np.delete(mat[i], i)
                scores.append(np.mean(row))
            return np.array(scores)

    nets = sorted(network_indices.keys())
    aec_net_scores = np.stack([aec_network_scores_for_subject(pc1[s]) for s in range(S)])  # (S, n_networks)
    aec_net_mean = np.nanmean(aec_net_scores, axis=0)
    aec_net_sd = np.nanstd(aec_net_scores, axis=0)
    top3_idx = np.argsort(aec_net_mean)[::-1][:3]

    # --- PSD metrics ---
    # Per-subject PSD band means across parcels/networks
    psd_band_means = []
    psd_broadband_mean = None
    psd_net_scores = []
    for s in range(S):
        _, psd_s = split_feats(pc1[s])
        psd_s = psd_s.reshape(n_entities, n_bands)  # (entities, bands)
        # Band means across entities
        psd_band_means.append(psd_s.mean(axis=0))
        # Network-level scores (average across non-gamma bands)
        try:
            gamma_idx = band_names.index("gamma")
        except ValueError:
            gamma_idx = None
        use_band_idx = [i for i in range(n_bands) if i != gamma_idx]
        if not avg_networks:
            # Aggregate parcels to network averages per subject
            per_net = []
            for net in nets:
                idx = np.asarray(network_indices[net], int)
                per_net.append(psd_s[idx][:, use_band_idx].mean())
            psd_net_scores.append(np.array(per_net))
        else:
            psd_net_scores.append(psd_s[:, use_band_idx].mean(axis=1))

    psd_band_means = np.stack(psd_band_means)  # (S, n_bands)
    psd_band_mean = (np.nanmean(psd_band_means, axis=0), np.nanstd(psd_band_means, axis=0))
    if "broadband" in band_names:
        bb_idx = band_names.index("broadband")
        psd_broadband_mean = (float(psd_band_mean[0][bb_idx]), float(psd_band_mean[1][bb_idx]))
    psd_net_scores = np.stack(psd_net_scores)  # (S, n_networks)
    psd_net_mean = np.nanmean(psd_net_scores, axis=0)
    psd_net_sd = np.nanstd(psd_net_scores, axis=0)
    psd_top2_idx = np.argsort(psd_net_mean)[::-1][:2]

    # --- Rank correlation of band-wise patterns for PCs 2..k vs PC1 (AEC) ---
    rho_mean = np.nan
    p_mean = np.nan
    if fc_bands and P > 1:
        # Vector over bands excluding gamma and broadband for stability
        band_vec_idx = [i for i, b in enumerate(band_names) if b not in {"gamma", "broadband"}]
        if band_vec_idx:
            # Base vector from PC1
            base_vec = []
            for s in range(S):
                aec_s, _ = split_feats(pc1[s])
                aec_s = aec_s.reshape(n_bands, aec_edges).mean(axis=1)
                base_vec.append(aec_s[band_vec_idx])
            base_vec = np.nanmean(np.stack(base_vec), axis=0)

            rhos = []
            ps = []
            for pc in range(P):
                if pc == pc_base:
                    continue
                vec_pc = []
                for s in range(S):
                    aec_s, _ = split_feats(importance_array[s, pc])
                    aec_s = aec_s.reshape(n_bands, aec_edges).mean(axis=1)
                    vec_pc.append(aec_s[band_vec_idx])
                vec_pc = np.nanmean(np.stack(vec_pc), axis=0)
                rho, pval = spearmanr(base_vec, vec_pc)
                rhos.append(rho)
                ps.append(pval)
            if rhos:
                rho_mean = float(np.nanmean(rhos))
                p_mean = float(np.nanmean(ps))

    # --- Printing ---
    print("Feature importance from latent occlusions.")
    print(f"Figure 6 shows mean absolute |z| changes induced by zeroing PC{pc_base+1} on {dataset.title()}.")

    # AEC section
    if fc_bands and aec_band_means is not None:
        mu, sd = aec_band_means
        def fmt(b):
            i = band_names.index(b)
            return f"{mu[i]:.3f} ± {sd[i]:.3f}"
        # Order and exclusion per prompt
        parts = [
            f"delta {fmt('delta')}",
            f"theta {fmt('theta')}",
            f"alpha {fmt('alpha')}",
            f"beta {fmt('beta')}",
            "gamma (excluded)",
            (f"broadband {fmt('broadband')}" if 'broadband' in band_names else None),
        ]
        parts = [p for p in parts if p is not None]
        print("AEC: Band-wise mean |z|: " + ", ".join(parts) + ".")
    else:
        print("AEC: Band-wise mean |z| not available (AEC computed broadband only).")

    # AEC network-wise
    n1, n2, n3 = nets[top3_idx[0]], nets[top3_idx[1]], nets[top3_idx[2]]
    v1 = f"{aec_net_mean[top3_idx[0]]:.3f} ± {aec_net_sd[top3_idx[0]]:.3f}"
    v2 = f"{aec_net_mean[top3_idx[1]]:.3f} ± {aec_net_sd[top3_idx[1]]:.3f}"
    v3 = f"{aec_net_mean[top3_idx[2]]:.3f} ± {aec_net_sd[top3_idx[2]]:.3f}"
    print(f"Network-wise, {n1} showed the highest importance ({v1}), followed by {n2} ({v2}) and {n3} ({v3}).")

    # PSD section
    if psd_broadband_mean is not None:
        print(f"PSD: Broadband mean |z| {psd_broadband_mean[0]:.3f} ± {psd_broadband_mean[1]:.3f};", end=" ")
    # PSD band-wise (excluding gamma)
    mu_psd, sd_psd = psd_band_mean
    def fmt_psd(b):
        i = band_names.index(b)
        return f"{mu_psd[i]:.3f} ± {sd_psd[i]:.3f}"
    parts = [
        f"delta {fmt_psd('delta')}",
        f"theta {fmt_psd('theta')}",
        f"alpha {fmt_psd('alpha')}",
        f"beta {fmt_psd('beta')}",
        "gamma (excluded)",
    ]
    print("band-wise: " + ", ".join(parts) + ".")

    pn1, pn2 = nets[psd_top2_idx[0]], nets[psd_top2_idx[1]]
    pv1 = f"{psd_net_mean[psd_top2_idx[0]]:.3f} ± {psd_net_sd[psd_top2_idx[0]]:.3f}"
    pv2 = f"{psd_net_mean[psd_top2_idx[1]]:.3f} ± {psd_net_sd[psd_top2_idx[1]]:.3f}"
    print(f"Highest regional importances were in {pn1} ({pv1}) and {pn2} ({pv2}).")

    if np.isfinite(rho_mean):
        print(f"Patterns were consistent for PCs 2–k (rank correlation of band-wise importance with PC1 = {rho_mean:.2f}, p {p_mean:.3f}).")

    print("Reporting tips: state mean ± SD across subjects, N = {},".format(S), end=" ")
    print("edge vs ROI aggregation = {} ({}), and multiple-comparison correction in Methods.".format(
        "networks" if avg_networks else "ROIs",
        "network-averaged" if avg_networks else "parcel-level",
    ))


if __name__ == "__main__":

    skip_compute = True

    dataset = "camcan"
    root_dir = f"/export01/data/{dataset}"
    model_path = f"/export01/data/{dataset}/saved_models/aefp/autoencoder/autoencoder_200.pt"

    emb_dir = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/{dataset}/ae")
    pca_path = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/interpretability/pca/{dataset}/latent_pca.joblib")
    explained_var = 0.95

    pca, scaler, _ = get_latent_pca(emb_dir, pca_path, explained_var)

    out_path = os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/interpretability/pc_occlusion/{dataset}/")
    fig_path = os.path.expanduser(f"~/dev/python/aefp/experiments/.figures/interpretability/pc_occlusion/{dataset}")
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    
    n_subjects = 100
    window_size = 900
    step_size = 1500

    max_pc = 50

    network_indices = get_network_indices(dataset, None)

    if skip_compute:
        # importances
        importances = np.load(os.path.join(out_path, f"pc_occlusion_importances_{n_subjects}subs.npy"))
        print(importances.shape)
        #plot_group_importances(importances, network_indices, fig_path, reduce_networks=True)

        #print_pc_occlusion_summary(importances, network_indices, dataset, pc_base=0, avg_networks=True, fc_bands=True)
        
        # errors
        errors = np.load(os.path.join(out_path, f"pc_occlusion_latent_errors_{n_subjects}subs_{max_pc}.npy"), allow_pickle=True)
        zocc_zhatocc, z_zhatocc, zrand_zhatocc = errors
        #zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z = errors
        #plot_latent_errors(zocc_zhatocc, zocc_zhat, z_zhatocc, zrand_zhatocc, zocc_z, fig_path)
        plot_latent_errors(zocc_zhatocc, z_zhatocc, zrand_zhatocc, fig_path)
    else:
    
        main(
            root_dir,
            model_path,
            out_path,
            fig_path,
            n_subjects,
            window_size,
            step_size,
            max_pc,
            network_indices,
        )

    ## plot icc
    pca_path = os.path.expanduser("~/dev/python/aefp/experiments/.tmp/interpretability/pca/camcan/latent_pca.joblib")
    train_iccs = plot_pca_icc(
        emb_dir=os.path.expanduser("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/camcan/ae/train"),
        pca_path=pca_path,
        out_dir=fig_path,
        max_components=300,
    )
    test_iccs = plot_pca_icc(
        emb_dir=os.path.expanduser("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/camcan/ae"),
        pca_path=pca_path,
        out_dir=fig_path,
        max_components=300,
    )