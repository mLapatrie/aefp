
import os
from glob import glob
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class MEGDataset(Dataset):

    def __init__(
        self,
        data_path,
        data_modality="meg", # "meg",
        data_type="rest", # "rest", "noise",
        data_space="source_200", # "sensor", "source_N", "source_N_sk"

        block_shape=(200, 900), # X channels, Y timepoints
        max_num_blocks=None, # Maximum number of blocks per subject

        sub_ids=None,
        val_percent=0.0,

        exclude_multiple_sessions=False, # If True, subjects with multiple sessions will be excluded

        verbose=True,

        device=torch.device("cpu")
    ):
        super().__init__()

        self.data_path = data_path
        self.data_modality = data_modality
        self.data_type = data_type
        self.data_space = data_space

        self.block_shape = block_shape
        self.max_num_blocks = max_num_blocks
        self.exclude_multiple_sessions = exclude_multiple_sessions

        self.sub_ids = self._get_sub_ids(sub_ids)
        self.val_percent = val_percent

        self.verbose = verbose

        self.device = device

        # Load data
        self.data = self._load_data(self.sub_ids, num_blocks=self.max_num_blocks)

        # Train/Validation split
        self.val_data = self.data[int(len(self.data) * (1 - self.val_percent)):] if self.val_percent > 0 else []
        self.train_data = self.data[:int(len(self.data) * (1 - self.val_percent))] if self.val_percent > 0 else self.data

        self.val_sub_ids = self.sub_ids[int(len(self.sub_ids) * (1 - self.val_percent)):] if self.val_percent > 0 else []
        self.train_sub_ids = self.sub_ids[:int(len(self.sub_ids) * (1 - self.val_percent))] if self.val_percent > 0 else self.sub_ids

        self.set_train()

    def set_train(self):
        self.data = self.train_data
        self.sub_ids = self.train_sub_ids

    def set_val(self):
        self.data = self.val_data
        self.sub_ids = self.val_sub_ids

    def _get_sub_ids(self, sub_ids_arg=None):
        sub_ids = sub_ids_arg
        if sub_ids_arg is None or type(sub_ids_arg) == int:
            sub_ids = sorted([i for i in os.listdir(self.data_path) if "sub" in i])
            print(f"Found {len(sub_ids)} subjects in {self.data_path}")

            # Filter out subjects without valid recordings
            sub_ids_filtered = [
                i for i in sub_ids 
                if len(glob(os.path.join(self.data_path, i, "ses-*", self.data_modality, self.data_type, self.data_space + ".pt"))) > 0
            ]
            print(f"Excluded {len(sub_ids) - len(sub_ids_filtered)} subjects without valid recordings")
            sub_ids = sub_ids_filtered
            
            # Exclude subjects with multiple sessions if specified
            if self.exclude_multiple_sessions:
                sub_ids_filtered = [
                    i for i in sub_ids_filtered 
                    if len(glob(os.path.join(self.data_path, i, "ses-*", self.data_modality, self.data_type, self.data_space + ".pt"))) == 1
                ]
                print(f"Excluded {len(sub_ids) - len(sub_ids_filtered)} subjects with multiple sessions")
            sub_ids = sub_ids_filtered

        np.random.shuffle(sub_ids)

        if type(sub_ids_arg) == int:
            sub_ids = sub_ids[:sub_ids_arg]
        
        return sub_ids
    
    def _load_data(self, sub_ids, num_blocks=None):
        if num_blocks is None:
            num_blocks = np.inf

        all_sub_blocks = []
        for sub_id in tqdm(sub_ids, desc="Loading subjects", disable=not self.verbose):
            ses_paths = glob(
                os.path.join(
                    self.data_path, 
                    sub_id, 
                    "ses-*", 
                    self.data_modality, 
                    self.data_type, 
                    self.data_space + ".pt"
                )
            )

            sub_ses_blocks = []
            for ses_path in ses_paths:
                ses_data = torch.load(ses_path, weights_only=False)

                # block the data
                num_channels, num_timepoints = ses_data.shape
                max_num_blocks = num_timepoints // self.block_shape[1]
                num_blocks = min(num_blocks, max_num_blocks)

                step = num_timepoints // num_blocks

                ses_blocks = []
                for block_idx in range(num_blocks):
                    start = block_idx * step
                    end = start + self.block_shape[1]
                    block_data = ses_data[:self.block_shape[0], start:end].to(self.device)

                    # Normalize block
                    block_data = (block_data - block_data.mean()) / block_data.std()

                    ses_blocks.append(block_data)
                sub_ses_blocks.append(ses_blocks)
            all_sub_blocks.append(sub_ses_blocks)
        
        return all_sub_blocks
    
    def __len__(self):
        return sum(len(ses_blocks) for sub_blocks in self.data for ses_blocks in sub_blocks)
    
    def __getitem__(self, idx, return_idx=False):
        sub_idx = 0
        ses_idx = 0
        block_idx = idx

        while block_idx >= len(self.data[sub_idx][ses_idx]):
            block_idx -= len(self.data[sub_idx][ses_idx])
            ses_idx += 1
            if ses_idx >= len(self.data[sub_idx]):
                sub_idx += 1
                ses_idx = 0
        
        if return_idx:
            return self.data[sub_idx][ses_idx][block_idx], (sub_idx, ses_idx, block_idx)
        return self.data[sub_idx][ses_idx][block_idx]
