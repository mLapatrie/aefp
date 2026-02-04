import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from aefp.utils.plotting_utils import style_axes, LABELS


def _infer_dataset_root_from_demographics_path(demographics_path: str) -> str | None:
    """Infer dataset root on disk from demographics file path.

    Returns one of '/export01/data/camcan' or '/export01/data/omega' if detected,
    otherwise None.
    """
    low = demographics_path.lower()
    if "camcan" in low:
        return "/export01/data/camcan"
    if "omega" in low:
        return "/export01/data/omega"
    return None


def _list_qc_subjects(dataset_root: str) -> set[str]:
    """List subject folder names under the dataset root (post-QC subjects).

    We assume subjects correspond to immediate subdirectories, e.g. 'sub-XXXXX'.
    If the directory cannot be read, returns an empty set.
    """
    try:
        entries = [d for d in os.listdir(dataset_root)
                   if os.path.isdir(os.path.join(dataset_root, d))]
        return set(entries)
    except Exception:
        return set()


def _build_id_aliases(ids: set[str]) -> set[str]:
    """Generate common alias forms for subject IDs to improve matching.

    Covers with/without 'sub-' prefix and '_' vs '-' variants.
    """
    aliases = set()
    for i in ids:
        forms = {i}
        # underscore ↔ hyphen variants
        forms.add(i.replace("_", "-"))
        forms.add(i.replace("-", "_"))
        # with/without sub- prefix
        if i.startswith("sub-"):
            base = i[len("sub-"):]
            forms.add(base)
            forms.add(base.replace("_", "-"))
            forms.add(base.replace("-", "_"))
        else:
            forms.add(f"sub-{i}")
            forms.add(f"sub-{i}".replace("_", "-"))
            forms.add(f"sub-{i}".replace("-", "_"))
        aliases.update(forms)
    return aliases


def load_embeddings(embeddings_dir: str) -> pd.DataFrame:
    """Load subject embeddings and average windows to obtain fingerprints."""
    files = sorted(glob(os.path.join(embeddings_dir, "sub-*.npy")))
    if not files:
        files = sorted(glob(os.path.join(embeddings_dir, "sub_*.npy")))
    fingerprints = []
    subjects = []
    for f in files:
        emb = np.load(f)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        subjects.append(os.path.splitext(os.path.basename(f))[0])
        fingerprints.append(emb)
    if not fingerprints:
        raise ValueError(f"No embeddings found in {embeddings_dir}")
    return pd.DataFrame(fingerprints, index=subjects)


def load_demographics(csv_path: str, id_col: str) -> pd.DataFrame:
    """Load demographics table indexed by subject identifier."""
    df = pd.read_csv(csv_path, sep='\t' if csv_path.endswith('.tsv') else None)
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not found in demographics file")
    df = df.set_index(id_col)

    # Replace CCID with sub-CCID
    if id_col == "CCID":
        df.index = [i.replace("_", "-") for i in df.index]

    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column in df matching any candidate (case-insensitive)."""
    # direct match first
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def print_demographics_summary(demog_df: pd.DataFrame, age_candidates: list[str], sex_candidates: list[str], label: str = "full dataset") -> None:
    """Print age stats and sex distribution for the provided demographics table."""
    if demog_df is None or demog_df.empty:
        print("No demographics available for summary.")
        return

    age_col = _find_column(demog_df, age_candidates + ["age"]) if age_candidates is not None else _find_column(demog_df, ["age"])
    sex_col = _find_column(demog_df, sex_candidates + ["sex", "gender"]) if sex_candidates is not None else _find_column(demog_df, ["sex", "gender"])

    print(f"\nDemographics summary ({label}):")

    # Age summary
    if age_col is not None:
        age = pd.to_numeric(demog_df[age_col], errors='coerce')
        age_non_null = age.dropna()
        if len(age_non_null) > 0:
            mean = float(age_non_null.mean())
            std = float(age_non_null.std(ddof=1)) if len(age_non_null) > 1 else 0.0
            a_min = float(age_non_null.min())
            a_max = float(age_non_null.max())
            print(f"  Age ({age_col}): {mean:.2f} ± {std:.2f} [min={a_min:.2f}, max={a_max:.2f}] (n={len(age_non_null)})")
        else:
            print(f"  Age ({age_col}): no non-missing values")
    else:
        print("  Age: column not found")

    # Sex/Gender distribution
    if sex_col is not None:
        sex_series = demog_df[sex_col].dropna()
        counts = sex_series.value_counts().to_dict()
        # Build a compact string like: value1=n1, value2=n2
        dist_str = ", ".join([f"{str(k)}={int(v)}" for k, v in counts.items()]) if counts else "no non-missing values"
        print(f"  {sex_col} distribution: {dist_str}")
    else:
        print("  Sex/Gender: column not found")


def run_regression(X: np.ndarray, y: np.ndarray, n_components: int, test_size: float, n_cv: int = 100):

    rng = np.random.RandomState(42)

    scores = []
    for _ in range(100):
        # shuffle data
        idxs = rng.permutation(len(X))
        X, y = X[idxs], y[idxs]

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # standardize
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # do PLS regression
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_test)

        score = r2_score(y_test, y_pred)
        scores.append(score)
    scores = np.array(scores)
    print(f"PLS regression R^2: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


def run_classification(X: np.ndarray, y: np.ndarray, test_size: float, n_cv: int = 100):
    rng = np.random.RandomState(42)

    scores = []
    for _ in tqdm(range(n_cv)):
        # shuffle data
        idxs = rng.permutation(len(X))
        X, y = X[idxs], y[idxs]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        scores.append( accuracy_score(y_test, clf.predict(X_test)))
    scores = np.array(scores)
    print(f"Logistic Regression accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


def density_jitter(y, center_x, max_width=0.20, seed=0):
    """Return x positions centered at center_x with jitter ∝ local 1D density at y."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return np.array([])
    # Silverman's rule-of-thumb bandwidth
    h = 1.06 * np.std(y, ddof=1) * (n ** (-1/5)) if n > 1 else 1.0e-3
    h = max(h, 1e-6)

    # Gaussian KDE evaluated at each point (O[n^2], fine for n≲1e3)
    diffs = (y[:, None] - y[None, :]) / h
    dens = np.exp(-0.5 * diffs**2).mean(axis=1) / (np.sqrt(2*np.pi) * h)

    # Scale widths by relative density (more density → more horizontal spread)
    w = max_width * (dens / dens.max())

    # Signed jitter within each point's width (slightly concentrated near edges avoids overplot)
    x = center_x + rng.uniform(-1.0, 1.0, size=n) * w
    return x

def plot_scatter(out_dir: str, fig_dir: str):
    print(out_dir)
    # --- helpers ---
    def load_scores(base_dir, subdir, sex_fname="Sex_scores.npy", age_fname="age_scores.npy"):
        d = base_dir if subdir is None else os.path.join(base_dir, subdir)
        sex_f = os.path.join(d, sex_fname)
        age_f = os.path.join(d, age_fname)
        if not (os.path.isfile(sex_f) and os.path.isfile(age_f)):
            return None
        acc_vals = np.load(sex_f)
        r_vals = np.load(age_f)  # keep sqrt like your original
        return acc_vals, r_vals

    def add_quartile_line(ax, x_pos, values):
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([x_pos, x_pos], [q25, q75], color="black", lw=2)
        ax.plot([x_pos - 0.1, x_pos + 0.1], [q50, q50], color="black", lw=2)

    # Your jitter function is assumed to exist already.
    # density_jitter(vals, center_x, max_width, seed)

    # Normalize out_dir so that if user passes ".../cont" we go up one level.
    out_dir = out_dir if out_dir.split("/")[-1] != "cont" else os.path.dirname(out_dir)

    # Which method -> subdirectory mapping (None means base folder)
    candidates = [
        ("PSD",  "psd"),
        ("AEC",  "aec"),
        ("AE",   "ae"),
        ("CONT", "cont"),
    ]

    # Load all available methods
    methods = []
    sex_arrays = []
    age_arrays = []
    for label, sub in candidates:
        loaded = load_scores(out_dir, sub)
        if loaded is None:
            # silently skip if that method's files don't exist
            continue
        acc_vals, r_vals = loaded
        methods.append(label)
        sex_arrays.append(acc_vals)
        age_arrays.append(r_vals)

    if len(methods) == 0:
        raise FileNotFoundError(
            f"No score files found under {out_dir} for any of {', '.join([c[0] for c in candidates])}."
        )

    # Sanity check: all arrays should have same length
    n_points = len(sex_arrays[0])
    for arr in sex_arrays + age_arrays:
        if len(arr) != n_points:
            raise ValueError("All score arrays must have the same length per method.")

    # --- print main results ---
    print("\nLoaded results summary (mean ± std):")
    for label, acc_vals, r_vals in zip(methods, sex_arrays, age_arrays):
        acc_mean, acc_std = float(np.mean(acc_vals)), float(np.std(acc_vals))
        r_mean, r_std = float(np.mean(r_vals)), float(np.std(r_vals))
        print(
            f"  {label}: Sex acc = {acc_mean:.3f} ± {acc_std:.3f} (n={len(acc_vals)}), "
            f"Age r² = {r_mean:.3f} ± {r_std:.3f} (n={len(r_vals)})"
        )

    # --- plotting ---
    M = len(methods)
    fig, ax1 = plt.subplots(figsize=(5.25, 5.25))
    ax2 = ax1.twinx()

    # Colors: use Matplotlib default cycle C0..C?
    colors = ["C3", "C2", "C0", "C1"]

    # X positions: [0..M-1] for Sex (left), [M..2M-1] for Age (right)
    xpos_sex = np.arange(M)
    xpos_age = np.arange(M, 2 * M)

    # Scatter with density jitter per method
    for i, (label, acc_vals, r_vals, color) in enumerate(zip(methods, sex_arrays, age_arrays, colors)):
        x_acc = density_jitter(acc_vals, center_x=xpos_sex[i], max_width=0.20, seed=1)
        x_r   = density_jitter(r_vals,   center_x=xpos_age[i], max_width=0.20, seed=1)

        # Classification (left axis)
        ax1.scatter(x_acc, acc_vals, color=color, alpha=0.8, label=label, s=25)
        add_quartile_line(ax1, xpos_sex[i], acc_vals)

        # Regression (right axis)
        ax2.scatter(x_r, r_vals, color=color, alpha=0.8, s=15)
        add_quartile_line(ax2, xpos_age[i], r_vals)

    # --- significance stars: only best vs second-best per metric ---
    def p_to_stars(p: float) -> str:
        return "*"

    def add_sig_bracket(ax, x1, x2, y, h, stars, color="black"):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c=color)
        ax.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", color=color)

    base_h = 0.02

    # Sex (classification) — pick best and second-best by mean
    sex_means = np.array([np.mean(v) for v in sex_arrays])
    order_sex = np.argsort(-sex_means)  # descending
    if len(order_sex) >= 2:
        i_best, i_second = int(order_sex[0]), int(order_sex[1])
        x1, x2 = xpos_sex[i_best], xpos_sex[i_second]
        # Ensure bracket spans left-to-right
        if x1 > x2:
            x1, x2 = x2 - 0.0, x1 + 0.0
        y_top = max(float(np.max(sex_arrays[i_best])), float(np.max(sex_arrays[i_second])))
        y = y_top + 0.05
        _, p = ttest_ind(sex_arrays[i_best], sex_arrays[i_second], equal_var=False)
        stars = p_to_stars(p)
        if stars:
            add_sig_bracket(ax1, x1, x2, y, base_h, stars)
            lo, hi = ax1.get_ylim()
            ax1.set_ylim(lo, max(hi, y + base_h + 0.03))

    # Age (regression) — pick best and second-best by mean
    age_means = np.array([np.mean(v) for v in age_arrays])
    order_age = np.argsort(-age_means)
    if len(order_age) >= 2:
        i_best, i_second = int(order_age[0]), int(order_age[1])
        x1, x2 = xpos_age[i_best], xpos_age[i_second]
        if x1 > x2:
            x1, x2 = x2 - 0.0, x1 + 0.0
        y_top = max(float(np.max(age_arrays[i_best])), float(np.max(age_arrays[i_second])))
        y = y_top + 0.02
        _, p = ttest_ind(age_arrays[i_best], age_arrays[i_second], equal_var=False)
        stars = p_to_stars(p)
        if stars:
            add_sig_bracket(ax2, x1, x2, y, base_h, stars)
            lo, hi = ax2.get_ylim()
            ax2.set_ylim(lo, max(hi, y + base_h + 0.03))

    # Vertical separator between sections
    sep_x = M - 0.5
    ax1.axvline(sep_x, color="k", linestyle="-", lw=1)

    # Axes labels/limits
    ax1.set_ylabel("Classification Accuracy (Sex)")
    ax2.set_ylabel("Ridge r² (Age)")
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_handles = [by_label[m] for m in methods]
    ordered_labels = [LABELS[m.lower()] for m in methods]
    ax1.legend(ordered_handles, ordered_labels, loc="lower left", bbox_to_anchor=(-0.035, -0.01), handletextpad=0.2, frameon=False)
    ax2.legend(ordered_handles, ordered_labels, loc="upper right", bbox_to_anchor=(1.015, 1.01), handletextpad=0.2, frameon=False)

    # X ticks & labels (repeat method names on both halves)
    xticks = np.concatenate([xpos_sex, xpos_age])
    xticklabels = methods + methods
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=0)
    ax1.set_xlim(-0.5, 2 * M - 0.5)

    # Section titles
    #ax1.text(0.25, -0.1, "Sex", ha="center", va="top", transform=ax1.transAxes, fontsize=11)
    #ax1.text(0.75, -0.1, "Age", ha="center", va="top", transform=ax1.transAxes, fontsize=11)

    # Baseline accuracy hline only under the Sex section
    baseline_acc = 0.5
    xmin, xmax = ax1.get_xlim()
    x0 = -0.5
    x1 = sep_x
    span = xmax - xmin
    ax1.axhline(baseline_acc,
                xmin=(x0 - xmin) / span,
                xmax=(x1 - xmin) / span,
                color="red", linestyle="--", lw=1)

    # Cosmetics
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "demographics_scatter.svg"))
    plt.close()


def main():

    skip_compute = False

    dataset = "camcan_sk"
    method = "psd"

    if dataset == "omega":
        embeddings_dir = os.path.expanduser(os.path.join("~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/omega/", f"{method}"))
        demographics = "/local01/data/raw_omega/participants.tsv"
        id_column = "participant_id"
        targets = ['sex', 'age']
    elif "camcan" in dataset:
        embeddings_dir = os.path.expanduser(os.path.join(f"~/dev/python/aefp/experiments/.tmp/interpretability/embeddings/{dataset}/", f"{method}"))
        demographics = "/export01/data/camcan/restricted_data.csv"
        id_column = "CCID"
        targets = ['Sex', 'age']
    
    out_dir = os.path.expanduser(os.path.join(f"~/dev/python/aefp/experiments/.tmp/interpretability/demographics_pls/{dataset}/", f"{method}"))
    fig_dir = os.path.expanduser(os.path.join(f"~/dev/python/aefp/experiments/.figures/interpretability/demographics_pls/{dataset}/"))
    os.makedirs(fig_dir, exist_ok=True)

    n_components = 2
    test_size = 0.25


    # Always load demographics, but summarize only for post-QC subjects
    demog_df = load_demographics(demographics, id_column)
    # Derive likely column names for age/sex from provided targets
    age_candidates = [t for t in targets if "age" in t.lower()] if targets else ["age"]
    sex_candidates = [t for t in targets if ("sex" in t.lower() or "gender" in t.lower())] if targets else ["sex", "gender"]

    # Infer dataset root and filter demographics to post-QC subjects
    dataset_root = _infer_dataset_root_from_demographics_path(demographics)
    qc_subjects = _list_qc_subjects(dataset_root) if dataset_root else set()
    qc_subjects_aliases = _build_id_aliases(qc_subjects) if qc_subjects else set()

    if qc_subjects_aliases:
        demog_qc = demog_df.loc[demog_df.index.intersection(qc_subjects_aliases)]
        if demog_qc.empty:
            print("Warning: No demographics matched post-QC subject directories; falling back to full dataset summary.")
            print_demographics_summary(demog_df, age_candidates, sex_candidates, label="full dataset (fallback)")
        else:
            removed = len(demog_df) - len(demog_qc)
            print(f"Found {len(demog_qc)} subjects after QC (removed {removed}).")
            print_demographics_summary(demog_qc, age_candidates, sex_candidates, label="post-QC subjects")
    else:
        print("Warning: Could not infer QC subjects from dataset root; printing full dataset summary.")
        print_demographics_summary(demog_df, age_candidates, sex_candidates, label="full dataset (fallback)")

    if not skip_compute:
        emb_df = load_embeddings(embeddings_dir)
        print(emb_df, demog_df)
        data = emb_df.join(demog_df, how="inner")
        if data.empty:
            raise RuntimeError("No overlapping subjects between embeddings and demographics")

        X = data[emb_df.columns].to_numpy()

        for target in targets:
            try:
                if target not in data.columns:
                    print(f"Target '{target}' not found, skipping")
                    continue
                y = data[target]
                X_target = X.copy()
                null_indices = y.isnull()
                if null_indices.any():
                    print(f"Removing {null_indices.sum()} subjects with missing values for target '{target}'")
                    X_target = X[~null_indices]
                    y = y[~null_indices]

                # shuffle
                rng = np.random.RandomState()
                idxs = rng.permutation(len(X_target))
                X_target = X_target[idxs]
                y = y[idxs]

                print(f"\n=== Target: {target} ===")
                if pd.api.types.is_numeric_dtype(y):
                    scores = run_regression(X_target, y.to_numpy(), n_components, test_size)
                else:
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y)
                    scores = run_classification(X_target, y_enc, test_size)
                
                out_path = os.path.join(out_dir, f"{target}_scores.npy")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                #np.save(out_path, scores)
                #print(f"Scores saved to {out_path}")

            except Exception as e:
                print(f"Error processing target '{target}': {e}")

    plot_scatter(os.path.dirname(out_dir), fig_dir)


if __name__ == "__main__":
    main()
