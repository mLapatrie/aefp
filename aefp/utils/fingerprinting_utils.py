
import os
import torch
import numpy as np

from glob import glob
from tqdm import tqdm

from aefp.utils.meg_utils import compute_aec_torch, compute_psd_torch


def fingerprint(X, correlation=False, mean_diff=True, verbose=False):
    # X of shape (n_subjects, 2, length_vector), where 2 represents the first and last precomputed vectors
    
    n_subjects = X.shape[0]
    
    confusion_matrix = np.zeros((n_subjects, n_subjects))
            
    for i in tqdm(range(n_subjects), desc="Computing confusion matrix", disable=not verbose):
        for j in range(n_subjects):
            confusion_matrix[i, j] = np.corrcoef(X[i, 0], X[j, 1])[0, 1] if correlation else np.linalg.norm(X[i, 0] - X[j, 1])
    
    # Find the minimum distance for each subject
    sel_subjects = np.argmax(confusion_matrix, axis=1) if correlation else np.argmin(confusion_matrix, axis=1)
    accuracy = np.sum(sel_subjects == np.arange(n_subjects)) / n_subjects
    
    # Find differentiability between subjects across each row
    differentiability = []
    for i in range(n_subjects):
        row = np.delete(confusion_matrix[i], i) # Remove self to not influence the distribution
        row_avg = np.mean(row)
        row_std = np.std(row)
        z_score = (confusion_matrix[i, i] - row_avg) / row_std
        differentiability.append(z_score)
    
    differentiability = np.array(differentiability)
    differentiability = np.mean(differentiability) if mean_diff else differentiability

    return accuracy, differentiability


def upper_aec_torch(x):
    """
    Extract the upper triangle of the AEC matrix, excluding the diagonal.
    """
    aec_matrix = compute_aec_torch(x)
    
    upper_triangle = []
    
    for i in range(aec_matrix.shape[0]):
        upper_triangle.append(aec_matrix[i][np.triu_indices(aec_matrix.shape[1], k=1)])
    
    return torch.stack(upper_triangle)


def flat_psd_torch(x):
    """
    Compute the PSD for each subject and flatten the output.
    """
    psd, _ = compute_psd_torch(x)
    
    # Flatten the PSD output to (batch_size, num_parcels * power_length)
    flat_psd = psd.view(psd.shape[0], -1)
    
    return flat_psd


def get_valid_test_subjects(
        sub_dict, 
        root_dir, 
        same_session=False,
        data_modality="meg",
        data_type="rest",
        data_space="source_200",
    ):
    # Get subject lists
    sub_ids = [sub for sub in os.listdir(root_dir) if sub.startswith("sub-")]
    train_sub_ids = sub_dict["train_sub_ids"]
    val_sub_ids = sub_dict["val_sub_ids"]
    test_sub_ids = [sub for sub in sub_ids if sub not in train_sub_ids and sub not in val_sub_ids]

    # Filter subjects based on the number of sessions
    if not same_session:
        # Filter test subjects to those with multiple sessions
        multiple_sessions = []
        for sub in test_sub_ids:
            pattern = os.path.join(root_dir, sub, "*", data_modality, data_type, f"{data_space}.pt")
            session_paths = glob(pattern)
            if len(session_paths) > 1:
                multiple_sessions.append(sub)
        test_sub_ids = [sub for sub in test_sub_ids if sub in multiple_sessions]
    else:
        # Filter test subjects to those with at least one session
        test_sub_ids = [sub for sub in test_sub_ids if len(glob(os.path.join(root_dir, sub, "*", data_modality, data_type, f"{data_space}.pt"))) > 0]
    
    return test_sub_ids, (sub_ids, train_sub_ids, val_sub_ids, test_sub_ids)
        