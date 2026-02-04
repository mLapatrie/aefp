
import torch
import numpy as np

from torch.utils.data import Dataset

from aefp.datasets.meg import MEGDataset


class ContrastiveDataset(Dataset):
    def __init__(
            self, 
            base_dataset, 
            positive_split=0.5,
            same_session=True, # If True, positive pairs are from the same session
            device=torch.device("cpu")
        ):
        super().__init__()
        self.base_dataset = base_dataset
        self.positive_split = positive_split
        self.same_session = same_session
        self.device = device

        self.train_sub_ids = base_dataset.train_sub_ids
        self.val_sub_ids = base_dataset.val_sub_ids

        if not isinstance(base_dataset, MEGDataset):
            print(type(base_dataset))
            raise TypeError("Base dataset type not implemented.")
        
        self.set_train()
    
    def set_train(self):
        self.base_dataset.set_train()
    
    def set_val(self):
        self.base_dataset.set_val()
    
    def _get_contrastive_pair(self, sub_idx, ses_idx, block_idx):

        dataset_array = self.base_dataset.data

        block1 = dataset_array[sub_idx][ses_idx][block_idx]
        
        y = 0
        if np.random.rand() < self.positive_split:
            y = 1
            # Positive pair
            other_ses_idx = ses_idx
            if not self.same_session:
                num_sessions = len(dataset_array[sub_idx])
                other_ses_idx = np.random.choice([i for i in range(num_sessions)])

            num_blocks = len(dataset_array[sub_idx][other_ses_idx])
            if num_blocks <= 1:
                raise ValueError("Not enough blocks in the session to create a positive pair.")

            other_block_idx = np.random.choice([i for i in range(num_blocks) if (i != block_idx or other_ses_idx != ses_idx)])
            block2 = dataset_array[sub_idx][other_ses_idx][other_block_idx]
        else:
            # Negative pair
            num_subjects = len(dataset_array)
            other_sub_idx = np.random.choice([i for i in range(num_subjects) if i != sub_idx])
            other_ses_idx = np.random.randint(len(dataset_array[other_sub_idx]))
            other_block_idx = np.random.randint(len(dataset_array[other_sub_idx][other_ses_idx]))

            block2 = dataset_array[other_sub_idx][other_ses_idx][other_block_idx]
        
        return block1, block2, y

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        block1, (sub_idx, ses_idx, block_idx) = self.base_dataset.__getitem__(idx, return_idx=True)
        _, block2, y = self._get_contrastive_pair(sub_idx, ses_idx, block_idx)

        return block1.to(self.device), block2.to(self.device), y, sub_idx