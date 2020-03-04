import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from cormorant.models import CormorantMD17
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, logging_printout, init_cuda, set_dataset_defaults
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():

    # Initialize arguments -- Just
    args = init_argparse('md17')

    # Initialize file paths
    all_files = init_file_paths(args.prefix, args.workdir, args.modeldir, args.logdir, args.predictdir, args.logfile, args.bestfile, args.checkfile, args.loadfile, args.predictfile)
    logfile, bestfile, checkfile, loadfile, predictfile = all_files
    args = set_dataset_defaults(args)

    # Initialize logger
    init_logger(args.logfile, args.log_level)

    # Write input paramaters and paths to log
    logging_printout(args)

    # Initialize device and data type
    device, dtype = init_cuda(args.cuda, args.dtype)

    # Initialize dataloader
    ntr, nv, nte, datasets, num_species, charge_scale = initialize_datasets(args.num_train, args.num_valid, args.num_test,
                                                                            args.datadir, args.dataset, subset=args.subset,
                                                                            force_download=args.force_download)
    args.num_train, args.num_valid, args.num_test = ntr, nv, nte

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}

    # Initialize model
    model = CormorantMD17(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
                        args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                        args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                        charge_scale, args.gaussian_mask,
                        args.top, args.input, args.num_mpnn_levels,
                        device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(optimizer, args.lr_init, args.lr_final, args.lr_decay,
                                               args.num_epoch, args.num_train, args.batch_size, args.sgd_restart,
                                               lr_minibatch=args.lr_minibatch, lr_decay_type=args.lr_decay_type)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()

if __name__ == '__main__':
    main()
