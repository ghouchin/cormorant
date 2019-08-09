import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.sparse import coo_matrix

import os
from itertools import islice
from math import inf

import logging

class KaggleTrainDataset(Dataset):
    """
    Data structure for a "Predicting Molecular Properties" dataset.  Extends PyTorch Dataset.
    Does not download or porcess. Based upon a pre-processed dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    """
    # TODO: Rewrite with a better format and using torch.tensors() instead of np.arrays()
    def __init__(self, data, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # Fix by hand for dataset
        included_species = torch.tensor([1, 6, 7, 8, 9])

        self.included_species = included_species

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None


    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        temp_data = {key: val[idx] for key, val in self.data.items()}

        data = {key: torch.from_numpy(temp_data[key]) for key in ['charges', 'positions']}

        data['one_hot'] = torch.from_numpy(temp_data['charges']).unsqueeze(-1) == self.included_species.unsqueeze(0)

        num_atoms = len(temp_data['charges'])

        rows, cols = temp_data['jj_edge'][:, 0], temp_data['jj_edge'][:, 1]
        jj_types = coo_matrix((temp_data['jj_type'], (rows, cols)), shape=(num_atoms, num_atoms)).todense()
        jj_values = coo_matrix((temp_data['jj_value'], (rows, cols)), shape=(num_atoms, num_atoms)).todense()

        data['jj_types'] = torch.from_numpy(jj_types)
        data['jj_values'] = torch.from_numpy(jj_values)

        return data
