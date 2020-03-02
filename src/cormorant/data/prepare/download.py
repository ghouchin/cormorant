import logging
import os

from cormorant.data.prepare.md17 import download_dataset_md17
from cormorant.data.prepare.qm9 import download_dataset_qm9
from cormorant.data.prepare.ase import download_dataset_ase

def prepare_dataset(datadir, dataset, subset=None, splits=None, cleanup=True, force_download=False, name=None, path=None):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    name : str, optional                                                                                                              
        If given, this is the name of the folder where the data will be stored for ase-db.                                                       
        Does nothing for other datasets.  
    paht : str, optional
        If given, this is the path to the ase-db that is being loaded.
        Does nothing for other datasets.
    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data. 

    Notes
    -----
    TODO: Delete the splits argument?
    """

    # If datasets have subsets,
    if subset:
        dataset_dir = [datadir, dataset, subset]
    elif name:
        dataset_dir = [datadir, dataset, name]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    # Assume one data file for each split
    datafiles = {split: os.path.join(
        *(dataset_dir + [split + '.npz'])) for split in split_names}

    # Check datafiles exist
    datafiles_checks = [os.path.exists(datafile)
                        for datafile in datafiles.values()]

    # Check if prepared dataset exists, and if not set flag to download below.
    # Probably should add more consistency checks, such as number of datapoints, etc...
    new_download = False
    if all(datafiles_checks):
        logging.info('Dataset exists and is processed.')
    elif all([not x for x in datafiles_checks]):
        # If checks are failed.
        new_download = True
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    # If need to download dataset, pass to appropriate downloader
    if new_download or force_download:
        logging.info('Dataset does not exist. Downloading!')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, splits, cleanup=cleanup)
        elif dataset.lower().startswith('md17'):
            download_dataset_md17(datadir, dataset, subset,
                                  splits, cleanup=cleanup)
        elif dataset.lower().startswith('ase-db'):
            download_dataset_ase(datadir, dataset, name, path, splits, cleanup=cleanup)
        else:
            raise ValueError(
                'Incorrect choice of dataset! Must chose qm9/md17!')

    return datafiles
