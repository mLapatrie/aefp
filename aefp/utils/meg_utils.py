
import torch
import numpy as np
import mne

from scipy.io import loadmat


DEFAULT_ANATOMY_PATH_CAMCAN = "/export01/data/camcan/@default/tess_cortex_pial_low.mat"
DEFAULT_ANATOMY_PATH_OMEGA = "/local01/data/subjects/fsaverage/label"


### PSD ###
def welch_torch(x, fs=150, nperseg=None, noverlap=None, window='hann'):
    
    if noverlap is None:
        noverlap = nperseg // 2 # Default overlap of 50%
    
    B, L = x.shape
    if L < nperseg:
        raise ValueError('Data length is too short for the selected nperseg')

    if window == "hann":
        win = torch.hann_window(nperseg, dtype=x.dtype, device=x.device)
    elif window == "hamming":
        win = torch.hamming_window(nperseg, dtype=x.dtype, device=x.device)
    else:
        raise ValueError("Invalid window type")
    
    step = nperseg - noverlap
    num_segments = (L - noverlap) // step

    Pxx = torch.zeros(nperseg // 2 + 1, dtype=x.dtype, device=x.device)

    for i in range(num_segments):
        segment = x[:, i*step:i*step+nperseg]

        segment_windowed = segment * win

        fft_segment = torch.fft.rfft(segment_windowed, n=nperseg)

        Pxx_segment = (fft_segment.real ** 2 + fft_segment.imag ** 2) / (fs * win.pow(2).sum())
        Pxx_segment = Pxx_segment.squeeze()

        Pxx += Pxx_segment / num_segments

    f = torch.fft.rfftfreq(nperseg, 1/fs)

    return f, Pxx


def compute_psd_torch(data, fs=150, log=False, fft=False):
    """
    Compute Welch and extract band power for each band for each ROI
    """

    nperseg = 2 * fs # Window length for welch
    
    batch_size, num_parcels, data_length = data.shape
    
    fft = fft or data_length < nperseg
    
    power_length = (data_length // 2 + 1) if fft else (int(nperseg / 2) + 1)
    powers = torch.zeros((batch_size*num_parcels, power_length), device=data.device) # If using fft, not actually power
    
    # Flatten the batch and parcels
    data = data.view(batch_size*num_parcels, data_length)

    for i in range(batch_size*num_parcels):
        
        # if length is smaller than 2 seconds (300 samples), use fft instead of welch
        if fft:
            f = torch.fft.rfftfreq(data.shape[1], 1/fs)
            Pxx = torch.abs(torch.fft.rfft(data[i])) ** 2
        else:
            f, Pxx = welch_torch(data[i].unsqueeze(0), fs=fs, nperseg=nperseg)
        
        if log:
            Pxx = torch.log(Pxx)
        
        Pxx = Pxx[:len(f)]
        
        powers[i][:len(f)] = Pxx

    # Reshape the powers to (batch_size, num_parcels, power_length)
    powers = powers.view(batch_size, num_parcels, power_length)
    
    return powers, f


### AEC ###
def hilbert_torch(x):
    N = x.shape[-1]

    transforms = torch.fft.fft(x, axis=-1)
    transforms[:, 1:N//2] *= -1j # positive frequency
    transforms[:, (N+2)//2 + 1:N] *= 1j # negative frequency
    transforms[:, 0] = 0 # zero out the DC component
    if N % 2 == 0:
        transforms[:, N//2] = 0
    
    return torch.fft.ifft(transforms, dim=-1)


def compute_aec_torch(x):

    batch_size = x.shape[0]

    batch_matrices = []

    for i in range(batch_size):

        analytic_signal = hilbert_torch(x[i])
        
        envelopes = torch.abs(analytic_signal)
        
        aec_matrix = torch.corrcoef(envelopes)

        batch_matrices.append(aec_matrix)

    aec_matrix = torch.stack(batch_matrices)
    
    return aec_matrix

    
def get_network_indices(dataset, anatomy_path=None):
    if dataset == "camcan":
        if anatomy_path is None:
            print(f"NO ANATOMY PATH PROVIDED, USING DEFAULT ANATOMY PATH, {DEFAULT_ANATOMY_PATH_CAMCAN}")
            anatomy_path = DEFAULT_ANATOMY_PATH_CAMCAN
            
        anatomy = loadmat(anatomy_path)
        atlases = anatomy["Atlas"][0]
        available_atlases = [atlas[0][0] for atlas in atlases]
        assert "Schaefer_200_17net" in available_atlases, "Schaefer200_17net not found in anatomy file"

        schaefer_atlas = atlases[available_atlases.index("Schaefer_200_17net")]
        parcel_labels = [parcel[3][0] for parcel in schaefer_atlas[1][0]]
        parcel_networks = [label.split("_")[0] for label in parcel_labels]

    if dataset == "omega":
        if anatomy_path is None:
            print(f"NO ANATOMY PATH PROVIDED, USING DEFAULT ANATOMY PATH, {DEFAULT_ANATOMY_PATH_OMEGA}")
            anatomy_path = DEFAULT_ANATOMY_PATH_OMEGA

        parc_name = "Schaefer2018_200Parcels_17Networks_order"
        subjects_dir = "/local01/data/subjects"
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
        parcel_labels = labels_lh + labels_rh

        parcel_networks = [label.name.split('_')[2] for label in parcel_labels]

    networks = np.unique(parcel_networks)
    network_indices = {network: [] for network in networks}
    for idx, network in enumerate(parcel_networks):
        network_indices[network].append(idx)

    return network_indices