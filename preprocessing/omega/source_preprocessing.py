import os
import mne
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import sleep
from pathlib import Path

# === USER SETTINGS ===
subjects_dir = "/local01/data/subjects"
bem_sol = subjects_dir + "/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif"

# Schaefer parcellation name — make sure it's installed in fsaverage/label
parc_name = "Schaefer2018_200Parcels_17Networks_order"

# Output dir
output_root = Path("/local01/data/omega")

# Preprocessing params
l_freq = 0.5
h_freq = 75.0
resample_sfreq = 150

def preprocess_session(session_path):
    # Load MEG run
    ds = sorted((session_path / "meg").glob("*rest*.ds"))[0]
    raw = mne.io.read_raw_ctf(ds, preload=True)

    # coregister
    coreg = mne.coreg.Coregistration(
        info=raw.info,
        subject='fsaverage',
        subjects_dir=subjects_dir,
    )

    # Fit using fiducials and headshape points
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=20, verbose=True)

    # Save transform
    coreg_trans = coreg.trans

    # Preprocessing
    raw.filter(l_freq, h_freq, fir_design='firwin')
    raw.notch_filter(freqs=[60, 120, 180], fir_design='firwin')
    raw.resample(resample_sfreq, npad="auto")
    tmax = min((45000-1)/150, (raw.n_times-1)/150)
    raw.crop(tmin=0, tmax=tmax)

    # Artifact removal (optional)
    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(
        raw, ch_name='ECG', n_grad=0, n_mag=2, n_eeg=0
    )
    eog_projs, _ = mne.preprocessing.compute_proj_eog(
        raw, ch_name='VEOG', n_grad=0, n_mag=2, n_eeg=0
    )
    raw.add_proj(ecg_projs + eog_projs)
    raw.apply_proj()

    # Setup source space on fsaverage
    src = mne.setup_source_space(
        'fsaverage', spacing='ico5', add_dist=False, subjects_dir=subjects_dir
    )
    bem = mne.read_bem_solution(bem_sol)

    # Compute forward
    fwd = mne.make_forward_solution(
        raw.info, trans=coreg_trans, src=src, bem=bem,
        meg=True, eeg=False
    )

    # Compute data covariance
    cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method='empirical')

    # LCMV
    filters = mne.beamformer.make_lcmv(
        raw.info,
        fwd,
        cov,
        reg=0.05,
        noise_cov=None,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank="info",
    )

    raw.pick_types(meg=True, eeg=False, stim=False, exclude='bads')
    stc = mne.beamformer.apply_lcmv_raw(raw, filters)

    print(f"STC shape: {stc.data.shape}")  # (n_vertices_total, n_times)

    # Read Schaefer-200 labels, skip "unknown"
    labels_lh = [
        lbl for lbl in mne.read_labels_from_annot(
            'fsaverage', parc=parc_name, hemi='lh', subjects_dir=subjects_dir
        ) if 'Medial_Wall' not in lbl.name
    ]
    labels_rh = [
        lbl for lbl in mne.read_labels_from_annot(
            'fsaverage', parc=parc_name, hemi='rh', subjects_dir=subjects_dir
        ) if 'Medial_Wall' not in lbl.name
    ]
    labels = labels_lh + labels_rh

    # Extract label time series
    label_ts = mne.extract_label_time_course(
        stc, labels, src, mode='mean_flip'
    )
    print(f"Label time series shape: {label_ts.shape}")  # (200, n_times)

    # Save
    subject = session_path.parts[-2]
    session = session_path.name
    out_dir = output_root / subject / session
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = torch.from_numpy(label_ts.astype('float32'))
    torch.save(tensor, out_dir / "source.pt")

    print(f"Saved: {out_dir / 'source.pt'}")
    print(f"Tensor of shape: {tensor.shape}")

def main():
    raw_root = Path("/local01/data/raw_omega")

    already_preprocessed = [
        sub for sub in os.listdir(output_root) if "sub" in sub
    ]

    #skip = ["sub-0197", "sub-0105"]
    skip = []
    print(f"Already preprocessed: {already_preprocessed}")

    for subject in raw_root.glob("sub-*"):
        if subject.name in already_preprocessed or subject.name in skip:
            print("skipping")
            continue
        for session in subject.glob("ses-*"):
            if (session / "meg").exists():
                try:
                    preprocess_session(session)
                except Exception as e:
                    with open("errors_source.txt", "a") as f:
                        f.write(f"{subject.name}/{session.name}\n")
                    continue

if __name__ == "__main__":
    main()

