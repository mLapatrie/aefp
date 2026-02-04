import os
import numpy as np
import torch
from glob import glob

from itertools import product
from tqdm import tqdm

from aefp.utils.utils import load_autoencoder
from aefp.utils.fingerprinting_utils import get_valid_test_subjects
from helpers.fingerprinting_helpers import (
    build_fp_array,
    compute_metrics,
    fit_minirocket_model,
)


def main(
        root_dir,
        model_path,
        model=None,
        sub_ids=None,
        method="ae", 
        data_modality="meg",
        data_type="rest",
        data_space="source_200",
        same_session=False, 
        window_size=4500,
        step_size=150,
        segment_size=900,
        segment_start=0,
        fs=150,
        band=None,
        num_subjects=100,
        bootstraps=1,
        bootstraps_size=0.9,
        save_dir=None,
        running_avg_X=None,
        running_avg_n=None,
        return_n=False,
        return_model=False,
        minirocket_num_features=10000,
        minirocket_max_dilations=32,
        minirocket_fit_subjects=20,
        minirocket_windows_per_subject=5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

    if model is None or sub_ids is None:
        # Load the model metadata for subject splits
        model, cfg, sub_dict = load_autoencoder(model_path, method=method)
        if method in {"ae", "ae_ft", "cont", "cross"}:
            model.to(device)
            model.eval()
        elif method == "minirocket":
            model = None

    if sub_ids is None:
        test_sub_ids, (sub_ids, train_sub_ids, val_sub_ids, _) = get_valid_test_subjects(
            sub_dict=sub_dict,
            root_dir=root_dir,
            same_session=same_session,
            data_modality=data_modality,
            data_type=data_type,
            data_space=data_space,
        )
        if "ft" in model_path:
            test_sub_ids = train_sub_ids
    else:
        test_sub_ids = sub_ids
        train_sub_ids = sub_ids

    subject_dirs = sorted([os.path.join(root_dir, sub) for sub in test_sub_ids])[:num_subjects]

    if method == "minirocket" and model is None:
        fit_sub_ids = train_sub_ids if train_sub_ids else test_sub_ids
        fit_dirs = sorted([os.path.join(root_dir, sub) for sub in fit_sub_ids])[:minirocket_fit_subjects]
        model = fit_minirocket_model(
            fit_dirs,
            window_size=window_size,
            step_size=step_size,
            segment_size=segment_size,
            segment_start=segment_start,
            data_modality=data_modality,
            data_type=data_type,
            data_space=data_space,
            same_session=same_session,
            band=band,
            fs=fs,
            max_subjects=minirocket_fit_subjects,
            max_windows_per_subject=minirocket_windows_per_subject,
            num_features=minirocket_num_features,
            max_dilations=minirocket_max_dilations,
            device=device,
        )

    X = build_fp_array(
        subject_dirs,
        model=model,
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
        save_windows_dir=save_dir,
        running_avg_X=running_avg_X,
        running_avg_n=running_avg_n,
        return_n=return_n,
        device=device,
    )
    if return_n:
        X, n = X

    # Compute metrics with optional bootstrapping over subjects
    results = []
    for _ in tqdm(range(int(bootstraps)), desc="Bootstrapping", leave=False):
        if bootstraps > 1:
            idx = np.random.choice(X.shape[0], size=int(X.shape[0] * bootstraps_size), replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        (acc_euc, diff_euc), (acc_cor, diff_cor) = compute_metrics(X_sample)
        results.append([acc_euc, np.mean(diff_euc), acc_cor, np.mean(diff_cor)])

    results = np.array(results)
    mean_res = results.mean(axis=0)
    std_res = results.std(axis=0)
    print(
        f"Fingerprinting accuracy (euclidean): mean={mean_res[0]:.4f}, std={std_res[0]:.4f}, Diff: mean={mean_res[1]:.4f}, std={std_res[1]:.4f}"
    )
    print(
        f"Fingerprinting accuracy (correlation): mean={mean_res[2]:.4f}, std={std_res[2]:.4f}, Diff: mean={mean_res[3]:.4f}, std={std_res[3]:.4f}"
    )

    if return_model:
        return results, X, n, (model, test_sub_ids)

    if return_n:
        return results, X, n
    return results, X


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    fs = 150
    window_size_sec = 6
    segment_size_sec = 120
    step_size_sec = 1

    window_size = window_size_sec * fs
    segment_size = segment_size_sec * fs
    step_size = 150#step_size_sec * fs

    dataset = "camcan"
    num_subjects = 100
    root_dir = f"/export01/data/{dataset}"
    save_dir = None #os.path.expanduser("~/dev/python/aefp/experiments/.tmp/fingerprint/")

    data_modality = "meg"
    data_type = "rest"
    data_space = "source_200"

    same_session = [True]
    bands = [None]
    method = ["cross"] # ["ae", "cont", "psd", "aec"]
    bootstraps = 100
    bootstraps_size = 0.9

    for m in method:
        
        if dataset == "camcan":
            model_path = "/export01/data/camcan/saved_models/aefp/autoencoder/autoencoder_200.pt"
            if m == "cont":
                model_path = "/export01/data/camcan/saved_models/aefp/encoder/encoder_200.pt"
            elif m == "cross":
                model_path = "/export01/data/camcan/saved_models/aefp/encoder/encoder_200_ce_1.pt"
        elif dataset == "omega":
            model_path = "/export01/data/omega/saved_models/aefp/autoencoder/autoencoder_200.pt"
            if m == "cont":
                model_path = "/export01/data/omega/saved_models/aefp/encoder/encoder_200.pt"
            elif m == "cross":
                model_path = "/export01/data/omega/saved_models/aefp/encoder/encoder_200_ce.pt"
        
        print(model_path)

        for ss in same_session:
            for b in bands:
                print(f"Running fingerprinting with method={m}, same_session={ss}, data_space={data_space}, band={b}")
                main(
                    root_dir=root_dir,
                    model_path=model_path,
                    method=m,
                    data_modality=data_modality,
                    data_type=data_type,
                    data_space=data_space,
                    same_session=ss,
                    window_size=window_size,
                    step_size=step_size,
                    segment_size=segment_size,
                    fs=fs,
                    band=b,
                    num_subjects=num_subjects,
                    bootstraps=bootstraps,
                    bootstraps_size=bootstraps_size,
                    save_dir=None,#save_dir,
                    device=device
                )
                print("Done.\n")
