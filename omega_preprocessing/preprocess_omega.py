import os
import mne
import torch
import numpy as np

from pathlib import Path
from time import sleep

def preprocess_session(session_path, out_root):
    # Load MEG run
    ds = sorted((session_path / "meg").glob("*rest*.ds"))[0]
    raw = mne.io.read_raw_ctf(ds, preload=True)

    # Preprocessing
    raw.filter(0.5, 74.99, fir_design='firwin')
    raw.notch_filter(freqs=[60, 120, 180], fir_design='firwin')
    raw.resample(150, npad="auto")

    # Artifact removal (optional)
    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(
        raw, ch_name='ECG', n_grad=0, n_mag=2, n_eeg=0
    )
    eog_projs, _ = mne.preprocessing.compute_proj_eog(
        raw, ch_name='VEOG', n_grad=0, n_mag=2, n_eeg=0
    )
    raw.add_proj(ecg_projs + eog_projs)
    raw.apply_proj()

    # Load template info
    template_channel_list = np.load("/local01/data/good_channels/channel_intersection.npy", allow_pickle=True).tolist()
    channel_list = [ch for ch in raw.ch_names if ch.split("-")[0] in template_channel_list]

    # Pick exact order from template
    raw.pick_channels(channel_list)

    # Extract data (fixed shape)
    data = raw.get_data()
    tensor = torch.from_numpy(data.astype('float32'))

    # Save
    subject = session_path.parts[-2]
    session = session_path.name
    out_dir = Path("/local01/data/omega") / subject / session
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_dir / "sensor_data.pt")
    print(f"Saved: {out_dir / 'sensor_data.pt'}")

def main():
    already_preprocessed = [sub for sub in os.listdir("/local01/data/omega") if "sub" in sub]
    print(already_preprocessed)

    raw_root = Path("/local01/data/raw_omega")
    for subject in raw_root.glob("sub-*"):
        if str(subject).split("/")[-1] in already_preprocessed:
            print("skip")
            continue
        for session in subject.glob("ses-*"):
            if (session / "meg").exists():
                try:
                    preprocess_session(session, raw_root)
                except:
                    continue

if __name__ == "__main__":
    main()

