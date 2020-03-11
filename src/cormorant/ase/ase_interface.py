from ase.calculators.calculator import Calculator
from cormorant.models import CormorantASE
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda, set_dataset_defaults
from cormorant.engine import init_optimizer, init_scheduler, rel_pos_deriv_to_forces
from cormorant.engine import Engine, ForceEngine
from cormorant.engine.engine import energy_and_force_mse_loss
# from cormorant.data.utils import initialize_datasets
from cormorant.data.prepare import gen_splits_ase
from cormorant.data.prepare.process import process_ase, process_db_row, _process_structure
from cormorant.data.collate import collate_fn
from cormorant.data.utils import convert_to_ProcessedDatasets
from torch.utils.data import DataLoader
import torch
import os
import logging
import numpy as np
from functools import partial
# from ase.db import connect


class ASEInterface(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model=None):
        Calculator.__init__(self)
        self.model = model

        self.edge_features = ['relative_pos']
        self.collate_fn = lambda x: collate_fn(x, edge_features=self.edge_features)

    @classmethod
    # def load(cls, filename, num_species, included_species=None):
    def load(cls, filename, num_species):
        saved_run = torch.load(filename)
        args = saved_run['args']
        charge_scale = max(included_species)

        # Initialize device and data type
        device, dtype = init_cuda(args.cuda, args.dtype)
        model = CormorantASE(*args, num_species=num_species, charge_scale=charge_scale,
                             device=device, dtype=dtype)
        model.load_state_dict(saved_run['model_state'])
        calc = cls(model)
        return calc

    def train(self, database, workdir=None, force_factor=0., num_epoch=256, lr_init=5.e-4, lr_final=5.e-6, batch_size=20):
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
        num_train = args.num_train
        #num_train = 10
        num_train, num_valid, num_test, datasets, num_species, max_charge = self.initialize_database(num_train, args.num_valid, args.num_test, args.datadir, database, force_train=force_train)

        # Construct PyTorch dataloaders from datasets
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         # num_workers=args.num_workers,
                                         num_workers=1,
                                         collate_fn=self.collate_fn)
                       for split, dataset in datasets.items()}

        # WE SHOULD NOT BE INITIALIZING THE MODEL HERE!!!
        if self.model is None:
            self.model = CormorantASE(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                                      args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                                      args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                                      max_charge, args.gaussian_mask,
                                      args.top, args.input, args.num_mpnn_levels,
                                      device=device, dtype=dtype)

        # Initialize the scheduler and optimizer
        optimizer = init_optimizer(self.model, args.optim, lr_init, args.weight_decay)
        scheduler, restart_epochs = init_scheduler(optimizer, lr_init, lr_final, args.lr_decay,
                                                   num_epoch, num_train, batch_size, args.sgd_restart,
                                                   lr_minibatch=args.lr_minibatch, lr_decay_type=args.lr_decay_type)


        if force_train:
            # Define a loss function.
            loss_fn = partial(energy_and_force_mse_loss, force_factor=force_factor)
            # Instantiate the training class 
            trainer = ForceEngine(args, dataloaders, self.model, loss_fn, optimizer, scheduler, args.target, restart_epochs,
                             bestfile=bestfile, checkfile=checkfile, num_epoch=num_epoch, 
                             num_train=num_train, batch_size=batch_size, device=device, dtype=dtype, uses_relative_pos=True,
                             save=args.save, load=args.load, alpha=args.alpha, lr_minibatch=args.lr_minibatch, predictfile=args.predictfile,
                             textlog=args.textlog)

        else:
            # Define a loss function. Just use L2 loss for now.
            loss_fn = torch.nn.functional.mse_loss
            # Instantiate the training class 
            trainer = Engine(args, dataloaders, self.model, loss_fn, optimizer, scheduler, args.target, restart_epochs,
                             bestfile=bestfile, checkfile=checkfile, num_epoch=num_epoch,
                             num_train=num_train, batch_size=batch_size, device=device, dtype=dtype,
                             save=args.save, load=args.load, alpha=args.alpha, lr_minibatch=args.lr_minibatch,
                             textlog=args.textlog)

        # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
        trainer.load_checkpoint()
        self.trainer = trainer

        # Train model.
        self.trainer.train()

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
        if not corm_input['relative_pos'].requires_grad and 'forces' in properties:
            print('calling force!')
            corm_input['relative_pos'].requires_grad_()

        energy = self.model(corm_input)
        self.results['energy'] = energy.detach().cpu().numpy()

        if 'forces' in properties:
            forces = self._get_forces(energy, corm_input)
            self.results['forces'] = forces.detach().cpu().numpy()[0][0]

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

        # If splits are not specified, automatically generate them.
        if splits is None:
            splits = gen_splits_ase(database, cleanup=True)

        # Set the number of points based upon the arguments
        num_pts = {'train': num_train, 'test': num_test, 'valid': num_valid}

        # Process ASE database, and return dictionary of splits
        ase_data = {}
        for split, split_idx in splits.items():
            ase_data[split] = process_ase(database, process_db_row, file_idx_list=split_idx, forcetrain=force_train)

        # Convert to Cormorant ProcessedDataset
        num_train, num_valid, num_test, datasets, num_species, max_charge = convert_to_ProcessedDatasets(ase_data, num_pts, subtract_thermo=False)
        self.included_species = datasets['train'].included_species

        return num_train, num_valid, num_test, datasets, num_species, max_charge

    def _get_forces(self, energy, batch):
        forces = []

        for i, pred in enumerate(energy):
            derivative_of_rel_pos = -torch.autograd.grad(pred, batch['relative_pos'], create_graph=True, retain_graph=True)[0]
            force = rel_pos_deriv_to_forces(derivative_of_rel_pos)
            forces.append(force)
        return torch.stack(forces, dim=0)

    def convert_atoms(self, atoms):
        data = _process_structure(atoms)
        # data['charges'] = data['charges'].unsqueeze(0)
        data = {key: val.unsqueeze(0) for key, val in data.items()}
        data['atom_mask'] = torch.ones(data['charges'].shape).bool()
        data['edge_mask'] = data['atom_mask'] * data['atom_mask'].unsqueeze(-1)
        data['one_hot'] = data['charges'].unsqueeze(-1) == self.included_species.unsqueeze(0).unsqueeze(0)
        return data




# def get_unique_images(database):
#     db = connect(database)
