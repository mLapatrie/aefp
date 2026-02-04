import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch

from helpers.interpretability_helpers import load_subject_embeddings, get_latent_pca
from aefp.utils.plotting_utils import style_axes

N_PCS = 30            # dimensionality for density/graph computations
K_DENS = 20           # neighbors for kNN density estimate
K_GRAPH = 20          # neighbors for geodesic graph
ALPHA = 0.9           # density penalty in edge cost: cost = dist * (1 + ALPHA * ell_avg)
SEG_SAMPLES = 160     # samples along centroid-to-centroid segment for valley depth
BINS = 180            # 2D display density grid
BLUR_SIGMA = 2.0      # for 2D display only
SUBSAMPLE_WINDOWS = None  # e.g. 400 to speed up; None = use all
RNG_SEED = 11

# ---------------------- Utilities ----------------------
_rng = np.random.default_rng(RNG_SEED)

import numpy as np
from sklearn.neighbors import NearestNeighbors
import heapq

def centroid_density_distance(
    emb,                    # (S, W, D)
    mode="graph",           # "graph" (kNN geodesic) or "segment" (line integral)
    k_dens=20,              # k for kNN density
    k_graph=20,             # k for graph neighbors (graph mode)
    alpha=0.9,              # density penalty: cost = dist * (1 + alpha * ell)
    seg_samples=160,        # #samples along line (segment mode)
    per_subject_windows=None,  # subsample per subject for speed (graph mode)
    random_state=7
):
    """
    Returns
    -------
    D : (S, S) ndarray
        Density-based distances between subject centroids.
    De : (S, S) ndarray
        Plain Euclidean distances between centroids (in the same D-space).

    Notes
    -----
    - 'ell' is a kNN log-density proxy: ell(x) ~ d * log r_k(x). Larger ell = sparser.
    - Graph mode builds an undirected kNN graph with edge weights:
        w_ij = ||xi - xj|| * (1 + alpha * 0.5*(ell_i + ell_j))
      and runs Dijkstra between centroid-representative nodes.
    - Segment mode approximates:
        D_ij ≈ ||Ci-Cj|| * mean_t (1 + alpha * ell( (1-t)Ci + tCj ))
    """
    rng = np.random.default_rng(random_state)
    S, W, D = emb.shape
    C = emb.mean(axis=1)  # centroids (S, D)

    # Euclidean centroid matrix (baseline)
    De = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)

    # Flatten windows; optionally subsample for the graph
    if mode == "graph":
        if per_subject_windows is not None and per_subject_windows < W:
            rows = []
            subj_of_row = []
            rep_idx_local = []
            for s in range(S):
                idx = rng.choice(W, size=per_subject_windows, replace=False)
                rows.append(emb[s, idx])             # (m, D)
                subj_of_row.extend([s]*len(idx))
                # for representative node, pick the subsampled one nearest the centroid
                diffs = emb[s, idx] - C[s]
                rep_idx_local.append(idx[np.argmin(np.linalg.norm(diffs, axis=1))])
            X = np.vstack(rows)                       # (N, D)
            subj_of_row = np.asarray(subj_of_row)     # (N,)
            # map representative row indices
            rep_rows = []
            cursor = 0
            for s in range(S):
                m = per_subject_windows
                block = X[cursor:cursor+m]
                j = np.argmin(np.linalg.norm(block - C[s], axis=1))
                rep_rows.append(cursor + j)
                cursor += m
            rep_rows = np.asarray(rep_rows, dtype=int)
        else:
            X = emb.reshape(-1, D)                    # (S*W, D)
            subj_of_row = np.repeat(np.arange(S), W)
            # representative = nearest window to centroid within each subject
            rep_rows = []
            for s in range(S):
                start = s*W
                block = X[start:start+W]
                j = np.argmin(np.linalg.norm(block - C[s], axis=1))
                rep_rows.append(start + j)
            rep_rows = np.asarray(rep_rows, dtype=int)

        # --- kNN log-density: ell(x) ~ d * log r_k(x) ---
        dens_nn = NearestNeighbors(n_neighbors=k_dens+1).fit(X)
        dists_k, _ = dens_nn.kneighbors(X, n_neighbors=k_dens+1, return_distance=True)
        rk = dists_k[:, -1]  # k-th neighbor radius (skip self)
        ell = X.shape[1] * np.log(np.clip(rk, 1e-12, None))  # (N,)

        # --- kNN graph (undirected) with density-weighted edges ---
        g_nn = NearestNeighbors(n_neighbors=k_graph+1).fit(X)
        dmat, nbrs = g_nn.kneighbors(X, n_neighbors=k_graph+1, return_distance=True)
        N = X.shape[0]
        adj_idx = [[] for _ in range(N)]
        adj_w   = [[] for _ in range(N)]
        for i in range(N):
            for d, j in zip(dmat[i, 1:], nbrs[i, 1:]):  # skip self (col 0)
                w_ij = float(d) * (1.0 + alpha * 0.5*(ell[i] + ell[j]))
                # undirected
                adj_idx[i].append(int(j)); adj_w[i].append(w_ij)
                adj_idx[j].append(int(i)); adj_w[j].append(w_ij)

        # --- Dijkstra per source centroid representative ---
        def dijkstra(src):
            dist = np.full(N, np.inf, dtype=np.float64)
            dist[src] = 0.0
            h = [(0.0, src)]
            while h:
                du, u = heapq.heappop(h)
                if du > dist[u]: 
                    continue
                for v, w in zip(adj_idx[u], adj_w[u]):
                    alt = du + w
                    if alt < dist[v]:
                        dist[v] = alt
                        heapq.heappush(h, (alt, v))
            return dist

        D = np.zeros((S, S), dtype=np.float64)
        # precompute all sources
        dist_from = [dijkstra(src) for src in rep_rows]
        for i in range(S):
            di = dist_from[i]
            for j in range(S):
                D[i, j] = di[rep_rows[j]]

        return D, De

    elif mode == "segment":
        # Fit density on all windows (no graph)
        X = emb.reshape(-1, D)
        dens_nn = NearestNeighbors(n_neighbors=k_dens+1).fit(X)
        def ell_query(Q):
            dq, _ = dens_nn.kneighbors(Q, n_neighbors=k_dens+1, return_distance=True)
            rk = dq[:, -1]
            return Q.shape[1] * np.log(np.clip(rk, 1e-12, None))

        # Straight-line integral with sampling
        Sidx = np.arange(S)
        Dseg = np.zeros((S, S), dtype=np.float64)
        for i in range(S):
            for j in range(i, S):
                if i == j:
                    Dseg[i, j] = 0.0
                    continue
                Ci, Cj = C[i], C[j]
                t = np.linspace(0, 1, seg_samples)
                P = (1 - t)[:, None] * Ci[None, :] + t[:, None] * Cj[None, :]
                ell_path = ell_query(P)
                length = np.linalg.norm(Cj - Ci)
                cost = (1.0 + alpha * ell_path).mean() * length
                Dseg[i, j] = Dseg[j, i] = cost
        return Dseg, De

    else:
        raise ValueError("mode must be 'graph' or 'segment'")


def plot_tsne_ax(ax, embeddings, distance_matrix, norm=None, perplexity=30, random_state=7, title=None):
    """t-SNE panel with no axis labels, ticks, or spines; centroids + colored edges."""
    S, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    subj_idx = np.repeat(np.arange(S), W)

    tsne = TSNE(perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(emb_flat).reshape(S, W, 2)
    Zc = Z.mean(axis=1)  # centroids (S,2)

    I, J = np.triu_indices(S, k=1)
    segs = np.stack([Zc[I], Zc[J]], axis=1)
    wts = distance_matrix[I, J]

    # points & centroids
    ax.scatter(Z.reshape(-1,2)[:,0], Z.reshape(-1,2)[:,1], s=10, c=subj_idx, cmap="tab20", alpha=0.6, lw=0)
    ax.scatter(Zc[:,0], Zc[:,1], s=50, c='black', marker='x', zorder=3)

    # colored centroid edges
    #lc = LineCollection(segs, cmap="viridis", alpha=0.6, norm=norm)
    #lc.set_array(wts)
    #lc.set_linewidth(2)
    #ax.add_collection(lc)

    # colorbar from the line collection
    #if cbar:
    #    mappable = cm.ScalarMappable(norm=lc.norm, cmap=lc.cmap)
    #    mappable.set_array(wts)
    #    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    # minimal styling: no labels, no ticks, no spines
    if title: ax.set_title(title, fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(False)
    #return lc

def plot_tsne_points_only_ax(ax, embeddings, perplexity=30, random_state=7, title=None, cmap_name="tab20"):
    S, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)

    tsne = TSNE(perplexity=perplexity, random_state=random_state, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(emb_flat).reshape(S, W, 2)

    cmap = plt.get_cmap(cmap_name)
    centroids = []

    for s in range(S):
        color = np.array(cmap(s % cmap.N))
        mid = W // 2
        face_color = color.copy(); face_color[-1] = 0.6
        edge_color = color.copy(); edge_color[-1] = 1.0

        if mid > 0:
            Z_first = Z[s, :max(0, mid-50)]
            if Z_first.size:
                ax.scatter(Z_first[:, 0], Z_first[:, 1], s=30,
                           facecolors=[face_color], edgecolors=[edge_color],
                           marker='o', linewidths=0)
        Z_second = Z[s, min(W, mid+50):]
        if Z_second.size:
            ax.scatter(Z_second[:, 0], Z_second[:, 1], s=30,
                       facecolors=[face_color], edgecolors=[edge_color],
                       marker='o', linewidths=0)

        c = Z[s].mean(axis=0)
        centroids.append(c)
        ax.scatter(c[0], c[1], color='black', marker='x', s=50, linewidths=1.5)

    centroids = np.array(centroids)

    # long arrow across the axes (bottom-left → top-right, slight randomness)
    x_min, x_max = ax.get_xlim(); y_min, y_max = ax.get_ylim()
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    rng = np.random.default_rng(random_state)
    theta = np.pi/4 + rng.uniform(-np.pi/12, np.pi/12)
    u = np.array([np.cos(theta), np.sin(theta)])

    # intersections with plot rectangle to span the panel
    ts = []
    for x_edge in (x_min, x_max):
        if u[0] != 0:
            t = (x_edge - center[0]) / u[0]
            y = center[1] + t * u[1]
            if y_min <= y <= y_max: ts.append(t)
    for y_edge in (y_min, y_max):
        if u[1] != 0:
            t = (y_edge - center[1]) / u[1]
            x = center[0] + t * u[0]
            if x_min <= x <= x_max: ts.append(t)
    t0, t1 = np.min(ts), np.max(ts)
    start = center + t0 * u
    end   = center + t1 * u

    arrow = FancyArrowPatch(posA=start, posB=end, arrowstyle='-|>', mutation_scale=12,
                            linewidth=1.8, color='black')
    ax.add_patch(arrow)
    #ax.text(end[0], end[1], "age", fontsize=11, ha='left', va='center')

    # orthogonal connectors computed in DISPLAY space
    to_disp = ax.transData.transform
    to_data = ax.transData.inverted().transform
    A = to_disp(start); B = to_disp(end)
    v = B - A; vv = np.dot(v, v)

    for c in centroids:
        C = to_disp(c)
        t = np.dot(C - A, v) / vv
        t = np.clip(t, 0.0, 1.0)
        F = A + t * v  # foot of perpendicular in display coords
        c_dat = c
        f_dat = to_data(F)
        ax.plot([c_dat[0], f_dat[0]], [c_dat[1], f_dat[1]],
                linestyle='--', color='black', linewidth=0.9)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_facecolor('none')

    return Z


def scatter_geo_vs_euc_ax(ax, De, Dgeo, title, vmin, vmax, marker="o", color="#7dd3fc", show_ylabel=True):
    """Bottom panels: points + best-fit; no legend; only left/bottom spines visible."""
    iu = np.triu_indices_from(De, k=1)
    x, y = De[iu], Dgeo[iu]

    ax.plot(x, y, marker, ms=3, alpha=0.6, color=color, linestyle="none")

    if x.size >= 2:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(vmin, vmax, 200)
        ax.plot(xx, m*xx + b, "--", lw=2, color="black", alpha=0.7)
        # bootstrap r^2
        n_bootstraps = 1000
        r2_bootstrap = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            sample_indices = np.random.choice(x.size, size=x.size, replace=True)
            r2_bootstrap[i] = np.corrcoef(x[sample_indices], y[sample_indices])[0, 1]**2
        ax.text(0.02, 0.98, rf"$r^2$={r2_bootstrap.mean():.3f} ± {r2_bootstrap.std():.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10)

    if not show_ylabel:
        ax.set_yticklabels([])
        #ax.set_yticks([])

    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Geodesic distance" if show_ylabel else "")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)

    # spines: keep only left & bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_fake_pca(
    S=5,            # subjects
    W=300,           # windows per subject
    D=30,            # dims (acts like 'after PCA')
    sep=8.0,         # inter-subject separation (ring radius in first 2 dims)
    sigma=1.0,       # within-subject std (isotropic)
    shared_shrink=0.8,   # bring shared centroids a bit closer together
    shared_bridge=0.20,  # fraction of shared windows replaced by mid-segment "bridges"
    bridge_jitter=0.5,   # jitter std for those bridges (in same units as sigma)
    seed=7
):
    """
    Returns
    -------
    emb_matched : (S, W, D) float32
    emb_same    : (S, W, D) float32
    """
    rng = np.random.default_rng(seed)

    # Subject centroids on a ring (dims 0-1); tiny noise on remaining dims
    angles = np.linspace(0, 2*np.pi, S, endpoint=False)
    baseC = np.zeros((S, D), dtype=np.float32)
    baseC[:, 0] = sep * np.cos(angles)
    baseC[:, 1] = sep * np.sin(angles)
    if D > 2:
        baseC[:, 2:] += rng.normal(0, 0.1, size=(S, D - 2)).astype(np.float32)

    # Matched: compact clusters around base centroids
    emb_matched = np.empty((S, W, D), dtype=np.float32)
    for s in range(S):
        emb_matched[s] = baseC[s] + rng.normal(0, sigma, size=(W, D))

    # Shared: centroids slightly closer + puff a bit + add bridges
    sharedC = (shared_shrink * baseC).astype(np.float32)
    emb_same = np.empty((S, W, D), dtype=np.float32)
    for s in range(S):
        emb_same[s] = sharedC[s] + rng.normal(0, 1.2 * sigma, size=(W, D))

    # Inject simple mid-segment bridges (only for shared)
    n_bridge = int(round(shared_bridge * W))
    if n_bridge > 0:
        for s in range(S):
            # choose random partner subjects and midpoints t in [0.25, 0.75]
            partners = rng.integers(0, S, size=n_bridge)
            partners = np.where(partners == s, (partners + 1) % S, partners)
            t = rng.uniform(0.25, 0.75, size=(n_bridge, 1))
            seg = (1 - t) * sharedC[s] + t * sharedC[partners]
            jitter = rng.normal(0, bridge_jitter * sigma, size=(n_bridge, D))
            bridges = seg + jitter
            # replace random indices within subject s
            idx = rng.choice(W, size=n_bridge, replace=False)
            emb_same[s, idx] = bridges.astype(np.float32)

    return emb_matched, emb_same


if __name__ == "__main__":
    # Legacy single-panel mode flag
    tsne_only = True

    no_regression = True

    use_fake_data = False

    n_windows_per_fp = 10
    n_sub_tsne = 10

    if not use_fake_data:
        dataset = "camcan"

        if dataset == "omega":
            embeddings_dir1 = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/omega/"))
            embeddings_dir2 = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/omega/"))

        elif dataset == "camcan":
            embeddings_dir1 = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings_step/camcan/"))
            embeddings_dir2 = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings_step/camcan_sk/"))
            embeddings_dir1_train = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/camcan/train"))
            embeddings_dir2_train = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/camcan_sk/train"))
        
        fig_dir = os.path.expanduser(os.path.join(f"~/dev/python/aefp/experiments/.figures/interpretability/tsne/{dataset}/"))
        os.makedirs(fig_dir, exist_ok=True)

        emb_matched, _ = load_subject_embeddings(embeddings_dir1)
        emb_same, _ = load_subject_embeddings(embeddings_dir2)

        max_windows = min(min([len(emb_matched[s]) for s in range(len(emb_matched))]), min([len(emb_same[s]) for s in range(len(emb_same))]))
        
        fp_matched = []
        fp_same = []
        for sub in range(len(emb_matched)):
            sub_fp = []
            for idx in range(0, max_windows - n_windows_per_fp, 3):
                sub_fp.append(emb_matched[sub][idx:idx + n_windows_per_fp].mean(axis=0))
            fp_matched.append(sub_fp)

        for sub in range(len(emb_same)):
            sub_fp = []
            for idx in range(0, max_windows - n_windows_per_fp, 3):
                sub_fp.append(emb_same[sub][idx:idx + n_windows_per_fp].mean(axis=0))
            fp_same.append(sub_fp)

        fp_matched = np.stack(fp_matched, axis=0)
        fp_same = np.stack(fp_same, axis=0)

        print(fp_matched.shape, fp_same.shape)

        # load pca
        pca, scaler, _ = get_latent_pca(
            emb_dir=[embeddings_dir1_train, embeddings_dir2_train],
            pca_path=os.path.expanduser(f"~/dev/python/aefp/experiments/.tmp/interpretability/pca/{dataset}/latent_pca_union.joblib"),
            explained_variance=30,
        )

        fp_matched_pca = pca.transform(scaler.transform(fp_matched.reshape(-1, fp_matched.shape[-1]))).reshape(fp_matched.shape[0], fp_matched.shape[1], -1)
        fp_same_pca = pca.transform(scaler.transform(fp_same.reshape(-1, fp_same.shape[-1]))).reshape(fp_same.shape[0], fp_same.shape[1], -1)

    else:
        fp_matched_pca, fp_same_pca = make_fake_pca(S=12, W=400, D=30, seed=42)

    np.random.seed(RNG_SEED)

    idx = np.random.choice(fp_matched_pca.shape[0], size=fp_matched_pca.shape[0], replace=False)
    fp_matched_pca = fp_matched_pca[idx]
    fp_same_pca = fp_same_pca[idx]
    if tsne_only:
        # Build 1×1 figure: legacy single stylized t-SNE (surrogate only)
        fig, ax = plt.subplots(figsize=(4, 6))

        plot_tsne_points_only_ax(
            ax,
            fp_same_pca[:n_sub_tsne],
            perplexity=30,
            random_state=RNG_SEED,
            title=None,
        )

        plt.savefig(os.path.join(fig_dir, f"tsne_points_{dataset}.svg"), transparent=True)
        plt.close()
    elif no_regression:
        # Build 1×2 figure: only the two top t-SNE panels
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)

        # t-SNE panels (distance_matrix not used by plot_tsne_ax; pass zeros for shape compatibility)
        S_top = fp_matched_pca[:n_sub_tsne].shape[0]
        dummy_D = np.zeros((S_top, S_top), dtype=float)

        plot_tsne_ax(
            ax_l,
            fp_matched_pca[:n_sub_tsne],
            distance_matrix=dummy_D,
            norm=None,
            perplexity=30,
            random_state=RNG_SEED,
            title="Native anatomy",
        )

        plot_tsne_ax(
            ax_r,
            fp_same_pca[:n_sub_tsne],
            distance_matrix=dummy_D,
            norm=None,
            perplexity=30,
            random_state=RNG_SEED,
            title="Surrogate anatomy",
        )

        plt.savefig(os.path.join(fig_dir, f"tsne_top_{dataset}.svg"))
        plt.close()
    else:
        Dgeo_matched, De_matched = centroid_density_distance(
            fp_matched_pca, mode="graph", k_dens=K_DENS, k_graph=K_GRAPH, alpha=ALPHA,
            seg_samples=SEG_SAMPLES, per_subject_windows=SUBSAMPLE_WINDOWS, random_state=RNG_SEED
        )
        distortion_matched = Dgeo_matched / De_matched
        distortion_matched[np.isnan(distortion_matched)] = 0.0 

        Dgeo_same, De_same = centroid_density_distance(
            fp_same_pca, mode="graph", k_dens=K_DENS, k_graph=K_GRAPH, alpha=ALPHA,
            seg_samples=SEG_SAMPLES, per_subject_windows=SUBSAMPLE_WINDOWS, random_state=RNG_SEED
        )
        distortion_same = Dgeo_same / De_same
        distortion_same[np.isnan(distortion_same)] = 0.0

        # Build 2×2 figure
        fig = plt.figure(figsize=(7, 5.25), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])  # ← top 2/3, bottom 1/3
        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1])
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_br = fig.add_subplot(gs[1, 1])

        vmin, vmax = 0, max(Dgeo_matched[:n_sub_tsne, :n_sub_tsne].max(), Dgeo_same[:n_sub_tsne, :n_sub_tsne].max())
        print(vmin, vmax)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Top row: t-SNEs (optionally subset subjects for speed/clarity)
        plot_tsne_ax(
            ax_tl,
            fp_matched_pca[:n_sub_tsne],
            distance_matrix=Dgeo_matched[:n_sub_tsne, :n_sub_tsne],  # distorsion
            norm=norm,
            perplexity=30,
            random_state=RNG_SEED,
            title="Native anatomy"
        )

        plot_tsne_ax(
            ax_tr,
            fp_same_pca[:n_sub_tsne],
            distance_matrix=Dgeo_same[:n_sub_tsne, :n_sub_tsne],  # distorsion
            norm=norm,
            perplexity=30,
            random_state=RNG_SEED,
            title="Surrogate anatomy"
        )

        # Bottom row: Geo vs Euc with best-fit (separate panels)
        Dgeo_vmin, Dgeo_vmax = min(Dgeo_matched.min(), Dgeo_same.min()), max(Dgeo_matched.max(), Dgeo_same.max())
        De_vmin, De_vmax = min(De_matched.min(), De_same.min()), max(De_matched.max(), De_same.max())
        lo_x, lo_y = 0.97 * De_vmin, 0.97 * Dgeo_vmin
        hi_x, hi_y = 1.03 * De_vmax, 1.03 * Dgeo_vmax
        scatter_geo_vs_euc_ax(ax_bl, De_matched, Dgeo_matched, vmin=lo_x, vmax=hi_x, title="", marker="o", color="dimgray")
        scatter_geo_vs_euc_ax(ax_br, De_same,    Dgeo_same, vmin=lo_x, vmax=hi_x,    title="", marker="o", color="dimgray", show_ylabel=False)

        ax_bl.set(xlim=(lo_x,hi_x), ylim=(lo_y,hi_y))
        ax_br.set(xlim=(lo_x,hi_x), ylim=(lo_y,hi_y))

        plt.savefig(os.path.join(fig_dir, f"tsne_{dataset}.svg"))
        plt.close()
