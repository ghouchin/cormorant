from ase.calculators.calculator import Calculator
from cormorant.models import CormorantASE
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda, set_dataset_defaults
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.engine import Engine
# from cormorant.data.utils import initialize_datasets
from cormorant.data.prepare import gen_splits_ase
from cormorant.data.prepare.process import process_ase, process_db_row
from cormorant.data.collate import collate_fn
from torch.utils.data import DataLoader
import torch
import os
import logging
import numpy as np
# from ase.db import connect


class ASEInterface(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, included_species):
        Calculator.__init__(self)
        self.model = model
        self.included_species = torch.tensor(included_species)

    @classmethod
    def load(cls, filename, num_species, included_species):
        saved_run = torch.load(filename)
        args = saved_run['args']
        charge_scale = max(included_species)

        # Initialize device and data type
        device, dtype = init_cuda(args.cuda, args.dtype)
        model = CormorantASE(*args, num_species=num_species, charge_scale=charge_scale,
                             device=device, dtype=dtype)
        model.load_state_dict(saved_run['model_state'])
        calc = cls(model, included_species)
        return calc

    def train(self, database, workdir=None, force_factor=0.):
        # This makes printing tensors more readable.
        torch.set_printoptions(linewidth=1000, threshold=100000)

        logging.getLogger('')

        # Initialize arguments -- Just
        args = init_argparse('ase-db')

        if workdir is not None:
            args.workdir = workdir

        # Initialize file paths
        all_files = init_file_paths(args.prefix, args.workdir, args.modeldir, args.logdir, args.predictdir, args.logfile, args.bestfile, args.checkfile, args.loadfile, args.predictfile)

        logfile, bestfile, checkfile, loadfile, predictfile = all_files
        args = set_dataset_defaults(args)

        # Initialize logger
        init_logger(logfile, args.log_level)

        # Initialize device and data type
        device, dtype = init_cuda(args.cuda, args.dtype)

        # Initialize dataloader
        force_train = (force_factor != 0.)
        ntr, nv, nte, datasets, num_species, charge_scale = self.initialize_database(args.num_train, args.num_valid, args.num_test, args.datadir, database, force_train=force_train)

        args.num_train, args.num_valid, args.num_test = ntr, nv, nte

        # Construct PyTorch dataloaders from datasets
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=args.num_workers,
                                         collate_fn=collate_fn)
                       for split, dataset in datasets.items()}

        model = CormorantASE(*args, device=device, dtype=dtype)

        # Initialize the scheduler and optimizer
        optimizer = init_optimizer(model, args.optim, args.lr_init, args.weight_decay)
        scheduler, restart_epochs = init_scheduler(optimizer, args.lr_init, args.lr_final, args.lr_decay,
                                                   args.num_epoch, args.num_train, args.batch_size, args.sgd_restart,
                                                   lr_minibatch=args.lr_minibatch, lr_decay_type=args.lr_decay_type)

        # Define a loss function. Just use L2 loss for now.
        loss_fn = torch.nn.functional.mse_loss

        # Instantiate the training class
        trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, args.target, restart_epochs,
                         bestfile=bestfile, checkfile=checkfile, num_epoch=args.num_epoch,
                         num_train=args.num_train, batch_size=args.batch_size, device=device, dtype=dtype,
                         save=args.save, load=args.load, alpha=args.alpha, lr_minibatch=args.lr_minibatch,
                         textlog=args.textlog)

        # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
        trainer.load_checkpoint()
        self.trainer = trainer

        # Train model.
        self.trainer.trainer()

        # Test predictions on best model and also last checkpointed model.
        self.trainer.evaluate()

    def calculate(self, atoms, properties, system_changes):
        """
        Populates results dictionary.

        Parameters
        ----------
        atoms : ASE Atoms object
            Atoms object from  ASE
        properties : list of strings
            Properties to calculate.
        system_changes : list
            list of what has changed.
        """
        Calculator.calculate(self, atoms)

        corm_input = self.convert_atoms(atoms)
        # Grad must be called for each predicted energy in the corm_input
        if not corm_input['positions'].requires_grad and 'forces' in properties:
            corm_input['positions'].requires_grad_()

        energy = self.model(corm_input)
        self.results['energy'] = energy

        if 'forces' in properties:
            forces = self._get_forces(energy, corm_input)
            self.results['forces'] = forces

    def initialize_database(self, num_train, num_valid, num_test, datadir, database, splits=None, force_train=True):
        """
        Initialized the ASE database into a format that the Pytorch routines can use

        Parameters
        ----------
        num_train: int
            Number of training points to use
        num_valid: int
            Number of validation points to use
        num_testing: int
            Number of testing points to use
        datadir
            path for where the data and will be stored
        database: str
            path to the ASE database
        Returns
        -------
        ntr, nv, nte, datasets, num_species, charge_scale

        """

        # Set the number of points based upon the arguments
        self.num_pts = {'train': num_train, 'test': num_test, 'valid': num_valid}

        # Download and process dataset. Returns datafiles.
        # datafiles = prepare_dataset(datadir, dataset, subset, splits, force_download=force_download, name=db_name, path=db_path)
        label = os.path.splitext(os.path.basename(database))[0]
        dataset_dir = [datadir, label]
        split_names = ['train', 'valid', 'test']

        # Assume one data file for each split
        datafiles = {split: os.path.join(
            *(dataset_dir + [split + '.npz'])) for split in split_names}

        # Check datafiles exist
        datafiles_checks = [os.path.exists(datafile)
                            for datafile in datafiles.values()]
        # Check if prepared dataset exists, and if not set flag to download below.
        # Probably should add more consistency checks, such as number of datapoints, etc...
        if all(datafiles_checks):
            logging.info('Dataset exists and is processed.')
        else:
            raise ValueError('All dataset files not found!  Make sure everything is in place.')

        # Define directory for which data will be output.
        asedir = os.path.join(*[datadir, label])

        # Important to avoid a race condition
        os.makedirs(asedir, exist_ok=True)

        logging.info('Downloading and processing the given ASE Database. Output will be in directory: {}.'.format(asedir))

        # If splits are not specified, automatically generate them.
        if splits is None:
            splits = gen_splits_ase(asedir, database, cleanup=True)

        # Process ASE database, and return dictionary of splits
        ase_data = {}
        for split, split_idx in splits.items():
            ase_data[split] = process_ase(database, process_db_row, file_idx_list=split_idx, stack=True, forcetrain=force_train)

        # Save processed ASE data into train/validation/test splits
        logging.info('Saving processed data:')
        for split, data in ase_data.items():
            savedir = os.path.join(asedir, split+'.npz')
            np.savez_compressed(savedir, **data)
        logging.info('Processing/saving complete!')

    def _get_forces(self, energy, batch):
        forces = []

        for i, pred in enumerate(energy):
            chunk_forces = -torch.autograd.grad(energy, batch['positions'], create_graph=True, retain_graph=True)[0]
            forces.append(chunk_forces[i])
        return torch.stack(forces, dim=0)

    def convert_atoms(self, atoms):
        data = {}
        atom_charges, atom_positions = [], []
        for i, line in enumerate(atoms.positions):
            atom_charges.append(atoms.numbers[i])
            atom_positions.append(list(line))
        atom_charges = torch.tensor(atom_charges).unsqueeze(0)
        atom_positions = torch.tensor(atom_positions).unsqueeze(0)
        data['charges'] = atom_charges
        data['positions'] = atom_positions
        data['atom_mask'] = torch.ones(atom_charges.shape).bool()
        data['edge_mask'] = data['atom_mask'] * data['atom_mask'].unsqueeze(-1)
        data['one_hot'] = data['charges'].unsqueeze(-1) == self.included_species.unsqueeze(0).unsqueeze(0)
        return data


# def get_unique_images(database):
#     db = connect(database)
