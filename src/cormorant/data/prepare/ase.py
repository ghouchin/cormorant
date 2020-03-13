import numpy as np
import torch

import logging
import os
import urllib

from os.path import join as join


from cormorant.data.prepare.process import process_ase, process_db_row
from cormorant.data.prepare.utils import cleanup_file

from ase.db import connect
import shutil 

def download_dataset_ase(datadir, dataname, name, path, splits=None, calculate_thermo=True, exclude=True, cleanup=True, force_train=False):
    """
    'Download' and prepare the given ase-db dataset.
    """
    # Define directory for which data will be output.
    asedir = join(*[datadir, dataname, name])

    # Important to avoid a race condition
    os.makedirs(asedir, exist_ok=True)

    logging.info(
        'Downloading and processing the given ASE Database. Output will be in directory: {}.'.format(asedir))

    # If splits are not specified, automatically generate them.
    path=path+name+'.db'
    if splits is None:
        print(asedir)
        print(path)
        splits = gen_splits_ase(path, cleanup)

    # Process ASE database, and return dictionary of splits
    ase_data = {}
    for split, split_idx in splits.items():
        ase_data[split] = process_ase(
            path, process_db_row, file_idx_list=split_idx, force_train=False)


    # Save processed ASE data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in ase_data.items():
        savedir = join(asedir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def gen_splits_ase(path, cleanup=True):
    """
    Generate ASE training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    logging.info('Splits were not specified! Automatically generating.')

    shutil.copy(path, 'temp.db')
    with connect('temp.db') as db:
        Nmols = db.count()
        index = np.linspace(1, Nmols, Nmols)

        for i in range(Nmols):
            row=db.get(id=Nmols-i)
            try:
                calc=row['calculator']
            except (KeyError, AttributeError):
                del db[Nmols-i]
                index=np.delete(index,Nmols-i-1)

        Nmols = db.count()

    Ntrain = int(0.8*Nmols)
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = index[train]
    valid = index[valid]
    test = index[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file('temp.db', cleanup)

    return splits


