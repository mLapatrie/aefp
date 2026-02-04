import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from aefp.utils.fingerprinting_utils import fingerprint, upper_aec_torch, flat_psd_torch


BANDS_DEF = {
    "broadband": (0.5, 45),
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def butter_bandpass(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    data = data.astype(np.float64)
    return filtfilt(b, a, data, axis=1)


def butter_bandstop(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="bandstop")
    data = data.astype(np.float64)
    return filtfilt(b, a, data, axis=1)


def occlude_feature_timeseries(
    data,
    feature_type,
    parcels,
    fs,
    shift,
    band=None,
):
    """Occlude an FC or PSD feature from a timeseries block.

    Parameters
    ----------
    data : np.ndarray
        Timeseries with shape ``(n_parcels, (n_seconds + shift) * fs)``.
    feature_type : {"fc", "psd"}
        Type of feature to occlude.
    parcels : tuple[int, int] or int
        Parcel indices defining the feature. For ``feature_type='fc'`` this
        should be a pair ``(i, j)``. For ``'psd'`` it is the single parcel
        index.
    fs : int
        Sampling frequency in Hz.
    shift : float
        Amount of time in seconds to shift when occluding FC features.
        ``data`` must contain ``shift`` extra seconds to allow the shift.
    band : str, optional
        Name of the frequency band to remove when ``feature_type='psd'``.

    Returns
    -------
    np.ndarray
        Occluded timeseries with shape ``(n_parcels, n_seconds * fs)`` where
        ``n_seconds`` equals ``data.shape[1] / fs - shift``.
    """

    shift_samples = int(shift * fs)
    n_samples = data.shape[1] - shift_samples
    if n_samples <= 0:
        raise ValueError("Shift is too large for the provided data block")

    out = data[:, :n_samples].copy()

    if feature_type == "fc":
        p1, p2 = parcels
        out[p2] = data[p2, shift_samples : shift_samples + n_samples]
    elif feature_type == "psd":
        if band is None:
            raise ValueError("band must be specified for PSD occlusion")
        lowcut, highcut = BANDS_DEF[band]
        filtered = butter_bandstop(data[parcels:parcels + 1], lowcut, highcut, fs)
        out[parcels] = filtered[0, :n_samples]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return out


def load_segments(
        subject_path, 
        data_modality="meg", 
        data_type="rest", 
        data_space="source_200", 
        segment_size=4500,
        segment_start=0,
        same_session=False,
        band=None,
        fs=150,
    ):
    """Return the first and last segments for a subject."""
    session_paths = sorted(glob(os.path.join(subject_path, "*", data_modality, data_type, f"{data_space}.pt")))
    if not same_session:
        session_paths = session_paths[:2]
        if len(session_paths) < 2:
            raise ValueError(f"Not enough session data found for subject {subject_path}")
        ses_0 = torch.load(session_paths[0])
        ses_1 = torch.load(session_paths[1])

        if band is not None:
            lowcut, highcut = BANDS_DEF[band]
            ses_0 = butter_bandpass(ses_0.numpy(), lowcut, highcut, fs=fs)
            ses_0 = torch.tensor(ses_0.copy(), dtype=torch.float32)

            ses_1 = butter_bandpass(ses_1.numpy(), lowcut, highcut, fs=fs)
            ses_1 = torch.tensor(ses_1.copy(), dtype=torch.float32)

        first_segment = ses_0[:, segment_start:segment_size+segment_start]
        last_segment = ses_1[:, segment_start:segment_size+segment_start]
    else:
        session_paths = [session_paths[0], session_paths[0]]
        ses_0 = torch.load(session_paths[0])
        if band is not None:
            lowcut, highcut = BANDS_DEF[band]
            ses_0 = butter_bandpass(ses_0.numpy(), lowcut, highcut, fs=fs)
            ses_0 = torch.tensor(ses_0.copy(), dtype=torch.float32)
        first_segment = ses_0[:, segment_start:segment_size+segment_start]
        last_segment = ses_0[:, -(segment_size+segment_start):]

    return first_segment, last_segment


def generate_sliding_windows(
        data,
        window_size=4500,
        step_size=150,
        segment_size=900,
    ):
    data = data[:, :segment_size]
    window_size = int(window_size)
    max_start = data.shape[1] - window_size
    starts = np.arange(0, max_start + 1, step_size).astype(int)

    windows = []
    for start in starts:
        win = data[:, start : start + window_size]
        win = (win - win.mean()) / win.std()
        
        windows.append(win)
    
    return windows


def _latent_from_window(batch, model, method, device):
    x = batch.unsqueeze(1).to(device)  # (1,1,C,window)
    match method:
        case "aec":
            out = upper_aec_torch(x.squeeze(1))
        case "psd":
            out = flat_psd_torch(x.squeeze(1))
        case "ae":
            with torch.no_grad():
                out = model.encode(x).sample()
        case "ae_ft":
            with torch.no_grad():
                out = model.encode(x).sample()
        case "cont":
            with torch.no_grad():
                out = model(x)
        case "cross":
            with torch.no_grad():
                out = model(x)
        case _:
            raise ValueError(f"Unknown method: {method}")
        
    if isinstance(out, np.ndarray):
        return out.reshape(-1)
    return out.cpu().detach().numpy().reshape(-1)


def process_subject(
    subject_path,
    model,
    method="ae",
    data_modality="meg",
    data_type="rest",
    data_space="source_200",
    same_session=False,
    window_size=900,
    step_size=150,
    segment_size=4500,
    segment_start=0,
    fs=150,
    band=None,
    save_windows_dir=None,
    running_avg_fp=None,
    running_avg_n=None,
    return_n=False,
    device="cuda",
):
    first_seg, last_seg = load_segments(
        subject_path, 
        data_modality=data_modality, 
        data_type=data_type, 
        data_space=data_space, 
        segment_size=segment_size, 
        segment_start=segment_start,
        same_session=same_session,
        band=band,
        fs=fs,
    )
    if first_seg is None or last_seg is None:
        return None

    fingerprints = []
    latent_windows = [[], []]
    for s_i, segment in enumerate([first_seg, last_seg]):
        # Generate sliding windows for the segment
        windows = generate_sliding_windows(
            segment,
            window_size,
            step_size,
            segment_size,
        )
        windows = torch.stack(windows)

        # Encode all windows
        n = 0 if running_avg_n is None else running_avg_n
        running_mean = None if running_avg_fp is None else running_avg_fp[s_i]
        latent_vectors = []
        for batch in windows.split(1):
            v = _latent_from_window(batch, model, method, device)
            if running_mean is None:
                running_mean = v
                n = 1
            else:
                n += 1
                running_mean += (v - running_mean) / n
            
            #if save_windows_dir is not None:
            latent_vectors.append(v)

        latent_vectors = np.stack(latent_vectors) if len(latent_vectors) > 0 else None

        fingerprints.append(running_mean)
        latent_windows[s_i].append(latent_vectors)

    if save_windows_dir is not None:
        os.makedirs(os.path.join(save_windows_dir, method), exist_ok=True)
        fname = f"{os.path.basename(subject_path)}_{segment_size}.npy"
        latent_windows = np.array(latent_windows).squeeze(1)
        latent_windows = latent_windows[:, :, :20000]
        np.save(os.path.join(save_windows_dir, method, fname), latent_windows)

    if return_n:
        return np.vstack(fingerprints), n
    return np.vstack(fingerprints)


def build_fp_array(subject_dirs, running_avg_X=None, return_n=False, *args, **kwargs):
    X = []
    for s, subject_path in enumerate(tqdm(subject_dirs, desc="Processing subjects", leave=False)):
        fp = process_subject(subject_path, running_avg_fp=running_avg_X[s] if running_avg_X is not None else None, return_n=return_n, *args, **kwargs)
        if return_n:
            fp, n = fp
        if fp is not None:
            fp = np.reshape(fp, (2, -1))
            X.append(fp)
    if return_n:
        return np.stack(X, axis=0), n
    return np.stack(X, axis=0)


def compute_metrics(X):
    acc_euc, diff_euc = fingerprint(X, correlation=False)
    acc_cor, diff_cor = fingerprint(X, correlation=True)
    return (acc_euc, diff_euc), (acc_cor, diff_cor)


def load_latent_vectors(latent_dir, method, segment_size):
    pattern = os.path.join(latent_dir, method, f"sub-*_{segment_size}.npy")
    files = sorted(glob(pattern))
    data = [np.load(f) for f in files]
    return np.stack(data)


def bootstrap_metrics(
    X,
    metric_idx: int = 2,
    bootstraps: int = 100,
    bootstraps_size: float = 0.9,
    return_all: bool = False,
):
    """Compute bootstrapped fingerprint metrics.

    Parameters
    ----------
    X : np.ndarray
        Array of fingerprints with shape (subjects, sessions, features).
    metric_idx : int, optional
        Which metric to return. Defaults to correlation accuracy (index 2).
    bootstraps : int, optional
        Number of bootstrap iterations. Defaults to 100.
    bootstraps_size : float, optional
        Fraction of subjects to sample in each bootstrap iteration.
    return_all : bool, optional
        If ``True`` return the array of bootstrap scores instead of the mean and
        standard deviation.
    """

    if bootstraps_size <= 1:
        idx_size = int(X.shape[0] * bootstraps_size)
    else:
        idx_size = int(bootstraps_size)

    results = []
    for _ in tqdm(range(int(bootstraps)), leave=False, desc="Bootstrapping metrics"):
        if bootstraps > 1:
            idx = np.random.choice(
                X.shape[0], size=idx_size, replace=False
            )
            X_sample = X[idx]
        else:
            X_sample = X
        (acc_euc, diff_euc), (acc_cor, diff_cor) = compute_metrics(X_sample)
        metric = [acc_euc, diff_euc, acc_cor, diff_cor][metric_idx]
        results.append(metric)

    results = np.array(results)
    if return_all:
        return results
    return results.mean(), results.std()


def compute_num_windows(segment_length, window_size, step_size):
    if segment_length < window_size:
        raise ValueError("Segment length must be >= window size")
    return int((segment_length - window_size) / step_size) + 1
