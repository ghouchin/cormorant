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

        self._split_jj_couplings()

        # At the moment, the only statistics we need are for the JJ couplings,
        # so we won't automatically calculate the rest of the statistics.
        self.stats = {}
        self._calc_jj_stats()


        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def _split_jj_couplings(self):
        jj_splits = {}
        jj_splits.update({'jj_'+str(key)+'_edge': [] for key in [1, 2, 3]})
        jj_splits.update({'jj_'+str(key)+'_value': [] for key in [1, 2, 3]})
        for jj_type, jj_edge, jj_value in zip(self.data['jj_type'], self.data['jj_edge'], self.data['jj_value']):
            for jj_t in [1, 2, 3]:
                this_jj = (jj_type == jj_t)

                jj_splits['jj_'+str(jj_t)+'_edge'].append(jj_edge[this_jj, :])
                jj_splits['jj_'+str(jj_t)+'_value'].append(jj_value[this_jj])

        self.data.update(jj_splits)

    def _calc_jj_stats(self):

        for jj_target, jj_split in self.data.items():

            if not jj_target in ['jj_1_value', 'jj_2_value', 'jj_3_value']: continue

            jj_values_cat = np.concatenate(jj_split)

            self.stats[jj_target[:4]] = (jj_values_cat.mean(), jj_values_cat.std())


    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        temp_data = {key: val[idx] for key, val in self.data.items()}

        data = {key: torch.from_numpy(temp_data[key]) for key in ['charges', 'positions']}

        data['one_hot'] = torch.from_numpy(temp_data['charges']).unsqueeze(-1) == self.included_species.unsqueeze(0)

        num_atoms = len(temp_data['charges'])

        for jj_type in [1, 2, 3]:

            jj_str = 'jj_'+str(jj_type)

            rows, cols = temp_data[jj_str+'_edge'][:, 0], temp_data[jj_str+'_edge'][:, 1]
            values = temp_data[jj_str+'_value']

            jj_values = coo_matrix((values, (rows, cols)), shape=(num_atoms, num_atoms)).todense()
            data[jj_str] = torch.from_numpy(jj_values)

        return data
