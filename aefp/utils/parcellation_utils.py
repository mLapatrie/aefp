
import os
import torch
import numpy as np

from scipy.io import loadmat


class Parcel:
    
    def __init__(
        self,
        parcel_vertices: torch.tensor,
        parcel_normals,
        parcel_label,
        reduction_function: torch,
    ):
        self.parcel_vertices = parcel_vertices
        self.parcel_normals = parcel_normals
        self.parcel_label = parcel_label
        self.reduction_function = reduction_function
        
        self.network = self.parcel_label.split("_")[0] # Only valid with Schaefer Yeo atlas
        
    def sign_flip(self, x):
        ### SIGN FLIP ###
        # Using the parcel_normals, find the main orientation of the parcel.
        # For every vertex in the parcel, check if the dot product of the vertex normal and the parcel normal is negative.
        # If it is, flip the sign of the vertex.
        
        parcel_normal = np.mean(self.parcel_normals, axis=0)
        
        for vert_idx in range(x.shape[0]):
            if np.dot(self.parcel_normals[vert_idx], parcel_normal) < 0:
                x[vert_idx] *= -1
                
        return x
    
    def __call__(self, x, do_sign_flip=True):
        # Only keep the vertices in the parcel.
        x = x[self.parcel_vertices]
        
        # Flip sign if asked.
        x = self.sign_flip(x) if do_sign_flip else x
        
        return self.reduction_function(x, axis=0)
    
    
class Atlas:
    
    def __init__(
        self,
        anatomy_path,
        parcellation_name,
        reduction_function="mean",
        do_sign_flip=True,
    ):
        self.anatomy_path = anatomy_path
        self.parcellation_name = parcellation_name
        
        if reduction_function == "mean":
            self.reduction_function = torch.mean
        elif reduction_function == "sum":
            self.reduction_function = torch.sum
        else:
            raise ValueError("Reduction function not recognized.")
        
        self.do_sign_flip = do_sign_flip
        
        self.parcels = self.load_parcellation()
        
    def load_parcellation(self):
        anatomy = loadmat(self.anatomy_path)
        atlases = anatomy["Atlas"][0]
        normals = anatomy["VertNormals"]
        
        available_atlases = [atlas[0][0] for atlas in atlases]
        if self.parcellation_name not in available_atlases:
            raise ValueError(f"Parcellation not found in default anatomy file.\nParcellations present in the file are: {available_atlases}")
        
        chosen_atlas = atlases[available_atlases.index(self.parcellation_name)]
        
        parcels = []
        for parcel in chosen_atlas[1][0]:
            parcel_vertices = torch.tensor((parcel[0][0] - 1).astype(np.int32))
            parcel_normals = normals[parcel_vertices]
            parcel_label = parcel[3][0]
            
            parcels.append(
                Parcel(
                    parcel_vertices,
                    parcel_normals,
                    parcel_label,
                    self.reduction_function
                )
            )
        return parcels
    
    def get_labels(self):
        return [parcel.parcel_label for parcel in self.parcels]
    
    def get_networks(self):
        if "Schaefer" in self.parcellation_name:
            return np.unique([parcel.network for parcel in self.parcels])
        else:
            return None
        
    def get_network_indices(self, network):
        return [idx for idx, parcel in enumerate(self.parcels) if parcel.network == network]
    
    def __call__(self, x):
        return torch.stack([parcel(x, self.do_sign_flip) for parcel in self.parcels])