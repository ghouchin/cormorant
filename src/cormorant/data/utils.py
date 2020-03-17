import torch
import numpy as np
import logging

from cormorant.data.dataset import ProcessedDataset
from cormorant.data.prepare import prepare_dataset


def initialize_datasets(num_train, num_valid, num_test, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False, db_name=None, db_path=None, force_train=False):
    """
    Initialize datasets.

    Parameters
    ----------
    num_train: int
        Number of training points to use
    num_valid: int
        Number of validation points to use
    num_testing: int
        Number of testing points to use
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    db-name : str, optional
        If given, this is the name of the folder where the data will be stored. 
        This will always be args.db-name if using ase-db but is shown here for clarity. 
        Does nothing for other datasets. 
    db-path : str, optional
        If given, this is the path to the ase-db that is being loaded.
        This will always be args.db-path if using ase-db but is shown here for clarity.
        Does nothing for other datasets.  

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': num_train,
               'test': num_test, 'valid': num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, subset, splits, force_download=force_download, name=db_name, path=db_path, force_train=False)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    num_train, num_valid, num_test, datasets, num_species, max_charge = convert_to_ProcessedDatasets(datasets, num_pts, subtract_thermo)
    # Basic error checking: Check the training/test/validation splits have the same set of keys.

    return num_train, num_valid, num_test, datasets, num_species, max_charge


def convert_to_ProcessedDatasets(datasets, num_pts, subtract_thermo=False):
    keys = [list(data.keys()) for data in datasets.values()]
    assert all(key == keys[0] for key in keys
               ), 'Datasets must have same set of keys!'

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    new_datasets = {}
    for split, data in datasets.items():
        pdset = ProcessedDataset(data, num_pts=num_pts.get(split, -1), included_species=all_species, subtract_thermo=subtract_thermo)
        new_datasets[split] = pdset
    datasets = new_datasets
    # datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    num_train = datasets['train'].num_pts
    num_valid = datasets['valid'].num_pts
    num_test = datasets['test'].num_pts
    return num_train, num_valid, num_test, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.

    """
    # Get a list of all species in the dataset across all splits
    split_species = {}
    for key, dataset in datasets.items():
        charges = dataset['charges']
        try:
            unique_charges = charges.unique(sorted=True)
        except AttributeError:
            unique_charges_per_item = [ci.unique(sorted=True) for ci in charges]
            unique_charges = torch.cat(unique_charges_per_item)
            unique_charges = unique_charges.unique(sorted=True)
        # If zero charges (padded, non-existent atoms) are included, remove them
        if unique_charges[0] == 0:
            unique_charges = unique_charges[1:]
        split_species[key] = unique_charges

    all_species = torch.cat([unique_i for unique_i in split_species.values()])
    all_species = all_species.unique(sorted=True)

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all(split.tolist() == all_species.tolist() for split in split_species.values()):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
