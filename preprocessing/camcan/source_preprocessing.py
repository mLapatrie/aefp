import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from aefp.utils.parcellation_utils import Atlas


def preprocess_subjects(
    data_path,
    atlas_instance,
    sensor_filename="sensor_data.pt",
    imaging_filename="imaging_kernel.pt", 
    interp_filename="interp_kernel.pt",
    output_filename="source_200.pt",
    block_size=4500,
    device=torch.device("cpu")
):
    """
    Generates source_200.pt files from sensor data and kernels.
    """
    
    # Find all subjects
    sub_ids = sorted([d for d in os.listdir(data_path) if "sub" in d and os.path.isdir(os.path.join(data_path, d))])
    print(f"Found {len(sub_ids)} subjects.")

    for sub_id in tqdm(sub_ids, desc="Processing subjects"):
        
        sub_dir = os.path.join(data_path, sub_id)
        rest_dir = os.path.join(sub_dir, "rest")
        
        sensor_path = os.path.join(rest_dir, sensor_filename)
        img_kernel_path = os.path.join(rest_dir, imaging_filename)
        
        # Interp kernel under sub_dir
        interp_kernel_path = os.path.join(sub_dir, interp_filename) 
        output_path = os.path.join(rest_dir, output_filename)

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        # Check required files exist
        if not (os.path.exists(sensor_path) and os.path.exists(img_kernel_path) and os.path.exists(interp_kernel_path)):
            print(f"Skipping {sub_id}: Missing input files.")
            continue

        try:
            # Load Data
            sensor_data = torch.load(sensor_path, weights_only=False).to(device)
            imagingk = torch.load(img_kernel_path, weights_only=False).to(device)
            interpk = torch.load(interp_kernel_path, weights_only=False).to(device)

            # Process in blocks to save RAM
            num_timepoints = sensor_data.shape[1]
            nblocks = (num_timepoints + block_size - 1) // block_size
            
            processed_blocks = []

            for i in range(nblocks):
                start = i * block_size
                end = min(start + block_size, num_timepoints)
                
                # 1. Slice Block
                block_sensor = sensor_data[:, start:end]

                # 2. Project to Individual Source Space
                # (Time, Sensors) @ (Sensors, Sources_Ind) -> (Time, Sources_Ind)
                source_ind = block_sensor.T @ imagingk.T

                # 3. Interpolate to Default Anatomy (Sparse MM)
                # (Sources_Def, Sources_Ind) @ (Sources_Ind, Time) -> (Sources_Def, Time)
                # Note: source_ind.T puts Time as second dimension
                source_def = torch.sparse.mm(interpk, source_ind.T)

                # 4. Parcellate using your Atlas
                # Result should be (N_Parcels, Time)
                parcellated_block = atlas_instance(source_def)
                
                processed_blocks.append(parcellated_block.cpu())

            # Concatenate all blocks along time dimension
            full_source_data = torch.cat(processed_blocks, dim=1)

            # Save
            torch.save(full_source_data, output_path)
            
        except Exception as e:
            print(f"Error processing {sub_id}: {e}")

if __name__ == "__main__":
    
    # CONFIGURATION
    DATA_PATH = "TODO"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Atlas
    default_anatomy = os.path.join(DATA_PATH, "@default/tess_cortex_pial_low.mat")

    atlas = Atlas(
        anatomy_path=default_anatomy,
        parcellation_name="Schaefer_200_17net",
        reduction_function="mean",
        do_sign_flip=True
    ).to(DEVICE)
    
    preprocess_subjects(DATA_PATH, my_atlas, device=DEVICE)