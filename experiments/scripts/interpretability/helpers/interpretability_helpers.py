
import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

from glob import glob
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib as mpl

from aefp.utils.meg_utils import compute_aec_torch, compute_psd_torch
from aefp.utils.meg_utils import get_network_indices

BANDS_DEF = {
    "broadband": (0.5, 45),
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def _mahalanobis_pca(z_pca_row, explained_var, eps: float = 1e-8):
    return float(np.sqrt(np.sum((z_pca_row ** 2) / (explained_var + eps))))


def butter_bandpass(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    data = data.astype(np.float64)
    return filtfilt(b, a, data, axis=1)


def load_latent_vectors(latent_dir, method, segment_size):
    pattern = os.path.join(latent_dir, method, f"sub-*_{segment_size}.npy")
    files = sorted(glob(pattern))
    data = [np.load(f) for f in files]
    return np.stack(data)


def load_subject_embeddings(emb_dir: str):
    files = sorted(glob(os.path.join(emb_dir, "sub-*.npy")))
    if not files:
        raise ValueError(f"No embeddings found in {emb_dir}")
    data = [np.load(f) for f in files]
    return data, files


def get_latent_pca(
    emb_dir: str,
    pca_path: str,
    explained_variance: float = 0.95,
):
    """Load or compute PCA for latent embeddings.

    Parameters
    ----------
    emb_dir : str
        Directory containing latent embeddings saved as ``sub-*.npy``.
    pca_path : str
        Path where the fitted PCA (and scaler) are stored via joblib.
    explained_variance : float, optional
        Desired amount of explained variance for PCA, by default ``0.95``.

    Returns
    -------
    tuple
        ``(pca, scaler)`` fitted on the embeddings.
    """

    if os.path.exists(pca_path):
        data = joblib.load(pca_path)
        return data["pca"], data["scaler"], data["maha_thresh"]

    print("Precomputed PCA not found, computing from embeddings...")

    if type(emb_dir) is str:
        embeddings, _ = load_subject_embeddings(emb_dir)
    elif type(emb_dir) is list:
        embeddings = []
        for d in emb_dir:
            emb, _ = load_subject_embeddings(d)
            embeddings.extend(emb)

    X = np.concatenate(embeddings, axis=0)
    X = X[::]
    print(X.shape)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=explained_variance)
    Z_train = pca.fit_transform(X_std)
    
    # compute mahalanobis threshold
    train_d2 = _mahalanobis_pca(Z_train, pca.explained_variance_, eps=1e-6)
    maha_thresh = np.percentile(train_d2, 95)

    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    joblib.dump({"pca": pca, "scaler": scaler, "maha_thresh": maha_thresh}, pca_path)

    return pca, scaler, maha_thresh


def icc_3_1(X):
    # X has shape (n_subjects, k_raters) where k_raters is 2 (beg and end vectors)

    n, k = X.shape
    mean_row = np.mean(X, axis=1, keepdims=True)
    mean_col = np.mean(X, axis=0, keepdims=True)
    grand = np.mean(X, keepdims=True)

    # sums of squares
    ssr = k * np.sum((mean_row - grand) ** 2)  # subjects
    ssc = n * np.sum((mean_col - grand) ** 2)  # raters
    sse = np.sum((X - mean_row - mean_col + grand) ** 2)  # residuals

    msr = ssr / (n - 1)  # mean square subjects
    msc = ssc / (k - 1)  # mean square raters
    mse = sse / ((n - 1) * (k - 1))  # mean square

    icc = (msr - mse) / (msr + (k - 1) * mse)
    return icc


def compute_parcel_scores(stats_path, dataset, anatomy_path, explained_var, use_aec=True, use_psd=True):
    """Compute parcel importance scores from feature statistics."""

    stats = np.load(stats_path, allow_pickle=True).item()
    slopes = stats["slopes"]
    # ``remove_pc_feature_stats.py`` yields a 2-D array (components, features)
    n_comp, n_feat = slopes.shape

    network_indices = get_network_indices(dataset, anatomy_path)
    n_parcels = sum(len(idx) for idx in network_indices.values())
    n_bands = len(BANDS_DEF)
    aec_len = n_parcels * (n_parcels - 1) // 2
    tri_idx = np.triu_indices(n_parcels, k=1)

    psd_len = n_parcels * n_bands
    assert n_feat == aec_len + psd_len, f"Expected {aec_len + psd_len} features, got {n_feat}"

    if explained_var is None:
        explained_var = np.ones(n_comp)
    explained_var = np.array(explained_var)
    explained_var = explained_var / explained_var.sum()

    scores = np.zeros(n_parcels)

    for c in range(n_comp):
        comp_slopes = np.abs(slopes[c])
        for p in range(n_parcels):
            idx_edges = np.where((tri_idx[0] == p) | (tri_idx[1] == p))[0]
            idx_bands = np.arange(p * n_bands, (p + 1) * n_bands)

            aec_score = comp_slopes[idx_edges].mean()
            psd_score = comp_slopes[idx_bands].mean()

            parcel_score = (aec_score * use_aec + psd_score * use_psd) / (use_aec + use_psd)
            scores[p] += explained_var[c] * parcel_score

    return scores


def normal_dist_p_value(mean1, std1, n1, mean2, std2, n2):
    # Compute standard error
    se = (std1**2 / n1 + std2**2 / n2) ** 0.5
    if se == 0:
        return 1.0 if mean1 == mean2 else 0.0

    # Compute t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = (std1**2 / n1 + std2**2 / n2) ** 2
    df_denom = ((std1**2 / n1) ** 2) / (n1 - 1) + ((std2**2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_denom

    # Two-sided p-value
    p = 2 * t.sf(abs(t_stat), df)
    return p


def extract_features(
    window: torch.Tensor,
    network_indices: dict,
    avg_networks_aec: bool = True,
    avg_networks_psd: bool = True,
    fc_bands: bool = True,
    n_parcels: int = 200,
    fs: int = 150,
) -> np.ndarray:
    """
    Build feature vector for one window:
      - AEC (full or per-band), optionally averaged over networks
      - PSD band means, optionally averaged over networks
    """
    device = window.device
    dtype = window.dtype
    x = window.unsqueeze(0)  # (1, P, T)
    bands = list(BANDS_DEF.values())

    # ---- helpers ----
    def to_idx_tensors(idx_map):
        return {k: torch.as_tensor(v, device=device, dtype=torch.long) for k, v in idx_map.items()}

    def upper_tri_flat(M):  # M: (..., N, N) -> (..., K)
        n = M.shape[-1]
        i, j = torch.triu_indices(n, n, offset=1, device=M.device)
        return M[..., i, j]

    def aec_for_signal(sig):  # sig: (1, P, T) -> (P, P)
        return compute_aec_torch(sig)[0]  # (P,P)

    def aec_block_mean(mat, idx_i, idx_j):
        return mat.index_select(0, idx_i).index_select(1, idx_j).mean()

    def aec_features():
        # compute AEC matrices: shape (B?, P, P) then optionally aggregate to networks
        if fc_bands:
            mats = []
            x_np = x[0].detach().cpu().numpy()
            for lo, hi in bands:
                xf = butter_bandpass(x_np, lo, hi, fs=fs).copy()
                xf = torch.as_tensor(xf, dtype=dtype, device=device).unsqueeze(0)
                mats.append(aec_for_signal(xf))
            aec = torch.stack(mats, dim=0)  # (B,P,P)
        else:
            aec = aec_for_signal(x)  # (P,P)

        if avg_networks_aec:
            nets = sorted(network_indices.keys())
            idx_map = to_idx_tensors(network_indices)
            n_net = len(nets)

            if fc_bands:
                out = torch.zeros((len(bands), n_net, n_net), device=device, dtype=dtype)
                for bi in range(len(bands)):
                    for i, ni in enumerate(nets):
                        for j, nj in enumerate(nets):
                            out[bi, i, j] = aec_block_mean(aec[bi], idx_map[ni], idx_map[nj])
                return upper_tri_flat(out).reshape(-1)
            else:
                out = torch.zeros((n_net, n_net), device=device, dtype=dtype)
                for i, ni in enumerate(nets):
                    for j, nj in enumerate(nets):
                        out[i, j] = aec_block_mean(aec, idx_map[ni], idx_map[nj])
                return upper_tri_flat(out)
        else:
            # parcel-level features
            if fc_bands:
                return upper_tri_flat(aec).reshape(-1)  # concat bands
            else:
                return upper_tri_flat(aec)

    def psd_features():
        psd, freqs = compute_psd_torch(x)  # psd: (1,P,F)
        psd = psd[0]  # (P,F)
        f = freqs.detach().cpu().numpy()

        # precompute band masks
        masks = [(f >= lo) & (f < hi) for lo, hi in bands]

        if avg_networks_psd:
            nets = sorted(network_indices.keys())
            idx_map = to_idx_tensors(network_indices)
            feats = []
            for ni in nets:
                mean_psd = psd.index_select(0, idx_map[ni]).mean(0).detach().cpu().numpy()
                for m in masks:
                    feats.append(mean_psd[m].mean())
            return torch.as_tensor(feats, device=device, dtype=dtype)
        else:
            feats = []
            psd_cpu = psd.detach().cpu().numpy()
            for p in range(psd_cpu.shape[0]):
                row = psd_cpu[p]
                for m in masks:
                    feats.append(row[m].mean())
            return torch.as_tensor(feats, device=device, dtype=dtype)

    # ---- build feature vector ----
    aec_feat = aec_features()
    psd_feat = psd_features()
    return torch.cat([aec_feat, psd_feat]).detach().cpu().numpy()


def rfft_freqs(n, fs, device):
    return torch.fft.rfftfreq(n, d=1.0/fs, device=device)  # (F,)

def bandpass_fft(x, fs, f_lo, f_hi):
    """
    Linear bandpass via FFT masking (diff'able).
    x: (B, P, T)
    """
    X = torch.fft.rfft(x, dim=-1)                          # (B,P,F)
    freqs = rfft_freqs(x.shape[-1], fs, x.device)          # (F,)
    mask = (freqs >= f_lo) & (freqs < f_hi)                # (F,)
    X = X * mask                                           # broadcast (B,P,F)
    y = torch.fft.irfft(X, n=x.shape[-1], dim=-1)          # (B,P,T)
    return y


def reconstruct_features(
    feature_values,
    network_indices,
    avg_networks: bool = True,
    fc_bands: bool = True,
):
    
    # 1. AEC features
    n = len(network_indices) if avg_networks else sum(len(idx) for idx in network_indices.values())
    aec_len = n * (n - 1) // 2
    tri_idx = np.triu_indices(n, k=1)

    if not fc_bands:
        aec_matrix = np.zeros((n, n), dtype=feature_values.dtype)
        aec_matrix[tri_idx[0], tri_idx[1]] = feature_values[:aec_len]
        aec_matrix += aec_matrix.T
        aec_matrix = aec_matrix.reshape(1, n, n)  # (1, n, n)
    else:
        n_bands = len(BANDS_DEF)
        aec_matrix = np.zeros((n_bands, n, n), dtype=feature_values.dtype)
        for b in range(n_bands):
            aec_matrix[b][tri_idx[0], tri_idx[1]] = feature_values[b * aec_len:(b + 1) * aec_len]
            aec_matrix[b] += aec_matrix[b].T
        aec_matrix = aec_matrix.reshape(n_bands, n, n)

    # 2. PSD features
    n = len(network_indices) if avg_networks else sum(len(idx) for idx in network_indices.values())
    n_bands = len(BANDS_DEF)
    psd_len = n * n_bands
    psd_features = np.zeros((n, n_bands), dtype=feature_values.dtype)
    for b in range(n_bands):
        psd_features[:, b] = feature_values[aec_len + b * n : aec_len + (b + 1) * n]

    return aec_matrix, psd_features


def plot_features(
    feature_values,
    network_indices,
    avg_networks: bool = True,
    fc_bands: bool = True,
    out_dir: str = ".",
):
    os.makedirs(out_dir, exist_ok=True)

    aec_matrix, psd_features = reconstruct_features(
        feature_values,
        network_indices,
        avg_networks,
        fc_bands,
    )
    psd_features = psd_features.T  # (n_bands, n)

    # expand psd features to number of parcels
    n_parcels = sum(len(idx) for idx in network_indices.values())
    psd_features_og = np.zeros((len(BANDS_DEF), n_parcels), dtype=psd_features.dtype)
    for i, (network, indices) in enumerate(network_indices.items()):
        psd_features_og[:, indices] = psd_features[:, i].reshape(-1, 1)
    psd_features = psd_features_og


    # AEC
    networks = sorted(network_indices.keys())
    n_networks = len(networks)

    # Plot six AEC matrices with a shared colorbar
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #vmax = np.nanmax(np.abs(np.stack(aec_matrix)))
    for ax, mat, band in zip(axes.ravel(), aec_matrix, BANDS_DEF.keys()):
        # remove lower triangle
        lower_tri = np.tril_indices(n_networks, k=0)
        mat[lower_tri] = np.nan

        im = ax.imshow(mat, cmap="Reds")#, vmin=0, vmax=vmax)
        ax.set_title(band)
        ax.set_xticks(range(n_networks))
        ax.set_yticks(range(n_networks))
        ax.set_xticklabels(networks, rotation=45, ha="right", fontsize="x-small")
        ax.set_yticklabels(networks, fontsize="x-small")
        ax.plot([-0.5, -0.5], [-0.5, 0.5], color="black", lw=1)
        for i in range(n_networks):
            ax.plot([i - 0.5, i - 0.5], [i - 0.5, i + 0.5], color="black", lw=1)
            ax.plot([i - 0.5, i + 0.5], [i + 0.5, i + 0.5], color="black", lw=1)
        ax.set_xlim(-0.5, n_networks - 0.5)
        ax.set_ylim(n_networks - 0.5, -0.5)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set left and bottom spines invisible
    for ax in axes.ravel():
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "aec_attribution.svg"), dpi=300)
    plt.show()


    # PSD
    tmp_dir = os.path.join(out_dir, "tmp_psd")
    att_dir = os.path.join(tmp_dir, "psd_attribution")
    os.makedirs(att_dir, exist_ok=True)

    band_names = list(BANDS_DEF.keys())
    for b, band in enumerate(band_names):
        pd.DataFrame(psd_features[b]).to_csv(
            os.path.join(att_dir, f"{band}.csv"), index=False
        )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    labels_path = os.path.join(repo_root, "Schaefer2018_200_17networks_labels.txt")
    r_script = os.path.join(repo_root, "plot_schaefer.R")

    try:
        subprocess.run(
            ["Rscript", r_script, labels_path, att_dir, os.path.join(out_dir, "psd_attribution"), "Reds"],
            check=True,
        )
    except Exception as e:
        print(f"Failed to run R script: {e}")
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def extract_features_diff(
    window: torch.tensor,
    network_indices: dict,
    avg_networks: bool = True,
    fc_bands: bool = True,
    device=None,
):
    tensor = window[0]

    networks = sorted(network_indices.keys())

    if avg_networks:
        n_networks = len(networks)

        # --- AEC ---
        if not fc_bands:
            aec_matrix = compute_aec_torch(tensor)[0]
            net_aec = torch.zeros((n_networks, n_networks), dtype=aec_matrix.dtype, device=device)
            for i, ni in enumerate(networks):
                idx_i = torch.tensor(network_indices[ni], device=device)
                for j, nj in enumerate(networks):
                    idx_j = torch.tensor(network_indices[nj], device=device)
                    net_aec[i, j] = aec_matrix[idx_i][:, idx_j].mean()

            tri_idx = torch.triu_indices(n_networks, n_networks, offset=1, device=device)
            aec_feat = net_aec[tri_idx[0], tri_idx[1]]
        
        else:
            net_aec_bands = torch.zeros((len(BANDS_DEF), n_networks, n_networks), dtype=tensor.dtype, device=device)
            for b, band in enumerate(BANDS_DEF.values()):
                f_lo, f_hi = band
                xb = bandpass_fft(tensor, fs=150, f_lo=f_lo, f_hi=f_hi)
                aec_matrix = compute_aec_torch(xb)[0]
                for i, ni in enumerate(networks):
                    idx_i = torch.tensor(network_indices[ni], device=device)
                    for j, nj in enumerate(networks):
                        idx_j = torch.tensor(network_indices[nj], device=device)
                        net_aec_bands[b, i, j] = aec_matrix[idx_i][:, idx_j].mean()

            tri_idx = torch.triu_indices(n_networks, n_networks, offset=1, device=device)
            aec_feat = net_aec_bands[:, tri_idx[0], tri_idx[1]].reshape(-1)


        # --- PSD ---
        psd, freqs = compute_psd_torch(tensor)
        psd = psd[0]  # (num_parcels, n_freqs)

        psd_feats = []
        for ni in networks:
            idx = torch.tensor(network_indices[ni], device=device)
            net_psd = psd[idx].mean(0)
            for band in BANDS_DEF.values():
                mask = (freqs >= band[0]) & (freqs < band[1])
                psd_feats.append(net_psd[mask].mean())

        psd_feat = torch.stack(psd_feats, dim=0)

        return torch.cat([aec_feat, psd_feat])
    

    # --- No network averaging ---
    n_parcels = sum(len(idx) for idx in network_indices.values())

    aec_matrix = compute_aec_torch(tensor)[0]
    tri_idx = torch.triu_indices(n_parcels, n_parcels, offset=1, device=device)
    aec_feat = aec_matrix[tri_idx[0], tri_idx[1]]

    psd, freqs = compute_psd_torch(tensor)
    psd = psd[0]

    psd_feats = []
    for parcel_idx in range(n_parcels):
        parcel_psd = psd[parcel_idx]
        for band in BANDS_DEF.values():
            mask = (freqs >= band[0]) & (freqs < band[1])
            psd_feats.append(parcel_psd[mask].mean())

    psd_feats = torch.stack(psd_feats)

    return torch.cat([aec_feat, psd_feats])


yeo_reduced_grouping = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 6, 6] # TempPar in Default
yeo_reduced_labels = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]

def reduce_by_groups_nan(
    A: np.ndarray,
    groups=yeo_reduced_grouping,
    m: int | None = None,
    exclude_diag: bool = False,
    reduction_type: str = "mean",
) -> np.ndarray:
    """
    Reduce an n×n matrix ``A`` into an m×m matrix using group labels,
    ignoring NaNs within each block. The reduction can be one of
    ``{"min", "max", "mean", "median"}``.

    Parameters
    ----------
    A : (n, n) array_like
        Input square matrix.
    groups : array_like of length n
        Group label for each row/column of ``A``.
    m : int or None
        Number of unique groups. If ``None``, inferred from ``groups``.
    exclude_diag : bool
        If True, within-group (i==j) blocks ignore diagonal entries.
    reduction_type : {"min", "max", "mean", "median"}
        Aggregation to apply inside each block. Defaults to "mean".

    Returns
    -------
    M : (m, m) ndarray
        Reduced block matrix with NaNs where a block has no valid entries.
    """
    A = np.asarray(A, dtype=float)
    g = np.asarray(groups)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != g.size:
        raise ValueError("A must be square n×n and len(groups)==n.")

    # Validate reduction type
    reduction_type = str(reduction_type).lower()
    allowed = {"min", "max", "mean", "median"}
    if reduction_type not in allowed:
        raise ValueError(f"reduction_type must be one of {sorted(allowed)}")

    uniq, inv = np.unique(g, return_inverse=True)
    if m is None:
        m = uniq.size
    if m != uniq.size:
        raise ValueError("m does not match the number of unique group labels.")

    # Pre-compute indices for each group (in the order of uniq)
    group_indices = [np.where(inv == k)[0] for k in range(m)]

    # Prepare output
    M = np.full((m, m), np.nan, dtype=float)

    # Select aggregation function
    if reduction_type == "mean":
        agg = np.nanmean
    elif reduction_type == "median":
        agg = np.nanmedian
    elif reduction_type == "min":
        agg = np.nanmin
    else:  # "max"
        agg = np.nanmax

    # Compute block-wise reduction
    with np.errstate(all="ignore"):
        for i in range(m):
            idx_i = group_indices[i]
            if idx_i.size == 0:
                continue
            for j in range(m):
                idx_j = group_indices[j]
                if idx_j.size == 0:
                    continue

                block = A[np.ix_(idx_i, idx_j)]

                # Exclude diagonal for within-group blocks if requested
                if exclude_diag and i == j and block.size > 0:
                    block = block.copy()
                    # Make diagonal NaN so it's ignored by nan-aggregations
                    np.fill_diagonal(block, np.nan)

                # If block has no elements after indexing, leave as NaN
                if block.size == 0:
                    M[i, j] = np.nan
                    continue

                # Apply aggregation; remains NaN if all values are NaN
                val = agg(block)
                M[i, j] = float(val) if np.size(val) == 1 else float(val.squeeze())

    return M


def plot_colorbar(
    OUT_PATH: str,
    WIDTH_IN: str,
    HEIGHT_IN: str,
    VMIN: str,
    VMAX: str,
    CMAP: str,
    ORIENTATION: str,
    TICK_SIDE: str,
):
    # cast to proper types
    width = float(WIDTH_IN)
    height = float(HEIGHT_IN)
    vmin = float(VMIN)
    vmax = float(VMAX)
    orientation = str(ORIENTATION)

    if vmax <= vmin:
        raise ValueError("VMAX must be greater than VMIN.")

    # colormap (get_cmap is deprecated; use matplotlib.colormaps)
    cmap = mpl.colormaps.get(CMAP)

    # figure and axes
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])  # fill most of the canvas

    # colorbar contents
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # draw colorbar
    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)

    # ticks: 5 evenly spaced, inclusive of vmin and vmax; no labels
    ticks = np.round(np.linspace(vmin, vmax, 4), 3)
    cbar.set_ticks(ticks)
    #cbar.set_ticklabels([""] * len(ticks))  # keep tick marks, hide numbers

    # optional: hide box
    #for spine in ax.spines.values():
    #    spine.set_visible(False)

    # place ticks on requested side
    if orientation == "vertical":
        if TICK_SIDE not in {"left","right"}:
            raise ValueError("TICK_SIDE must be 'left' or 'right' for vertical.")
        cbar.ax.yaxis.set_ticks_position(TICK_SIDE)
        cbar.ax.yaxis.set_label_position(TICK_SIDE)
    else:  # horizontal
        if TICK_SIDE not in {"top","bottom"}:
            raise ValueError("TICK_SIDE must be 'top' or 'bottom' for horizontal.")
        cbar.ax.xaxis.set_ticks_position(TICK_SIDE)
        cbar.ax.xaxis.set_label_position(TICK_SIDE)

    fig.savefig(OUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)