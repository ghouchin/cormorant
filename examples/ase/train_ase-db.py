import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from cormorant.models import CormorantQM9
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda, set_dataset_defaults
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets

from cormorant.data.collate import collate_fn

# This is a script adopted from train_qm9.py to train user input data from as ASE Database!


# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def main():

    # Initialize arguments -- Just
    args = init_argparse('ase-db')

    # Initialize file paths
    all_files = init_file_paths(args.prefix, args.workdir, args.modeldir, args.logdir, args.predictdir, args.logfile, args.bestfile, args.checkfile, args.loadfile, args.predictfile)
    logfile, bestfile, checkfile, loadfile, predictfile = all_files
    args = set_dataset_defaults(args)

    # Initialize logger
    init_logger(logfile, args.log_level)

    # Initialize device and data type
    device, dtype = init_cuda(args.cuda, args.dtype)

    # Initialize dataloader

    ntr, nv, nte, datasets, num_species, charge_scale = initialize_datasets(args.num_train, args.num_valid, args.num_test, args.datadir,
                                                                            'ase-db', db_name=args.db_name, db_path=args.db_path,
                                                                            force_download=args.force_download
                                                                            )
    args.num_train, args.num_valid, args.num_test = ntr, nv, nte

    #qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    # for dataset in datasets.values():
    #    dataset.convert_units(qm9_to_eV)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                   for split, dataset in datasets.items()}

    # Initialize model
    model = CormorantQM9(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                         args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                         args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                         charge_scale, args.gaussian_mask,
                         args.top, args.input, args.num_mpnn_levels,
                         device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(model, args.optim, args.lr_init, args.weight_decay)
    scheduler, restart_epochs = init_scheduler(optimizer, args.lr_init, args.lr_final, args.lr_decay,
                                               args.num_epoch, args.num_train, args.batch_size, args.sgd_restart,
                                               lr_minibatch=args.lr_minibatch, lr_decay_type=args.lr_decay_type)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    #cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, args.target, restart_epochs,
                     bestfile=bestfile, checkfile=checkfile, num_epoch=args.num_epoch,
                     num_train=args.num_train, batch_size=args.batch_size, device=device, dtype=dtype,
                     save=args.save, load=args.load, alpha=args.alpha, lr_minibatch=args.lr_minibatch,
                     textlog=args.textlog)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()


if __name__ == '__main__':
    main()
