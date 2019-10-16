import torch
from torch.utils.data import DataLoader

import logging

from cormorant.models import EdgeCormorant
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils_kaggle import init_nmr_kaggle_dataset

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def main():

    # Initialize arguments -- Just
    args = init_argparse('qm9')

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    # additional_atom_features = ['G_charges']
    # additional_atom_features = ['partial_qs']
    additional_atom_features = None

    ##### DEBUG #####
    top = 'linear'
    input_type = 'linear'
    mpnn_levels = 0
    num_top_levels = 0
    top_activation = 'tanh'
    #################

    args, datasets, num_species, charge_scale = init_nmr_kaggle_dataset(args, args.datadir, file_name='targets_train_expanded.npz', additional_atom_features=additional_atom_features)
    edge_features = ['jj_1', 'jj_2', 'jj_3', '1JHC', '1JHN', '2JHH', '2JHC', '2JHN', '3JHH', '3JHC', '3JHN', 'jj_all']

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=lambda x: collate_fn(x, edge_features)) for split, dataset in datasets.items()}

    # Initialize model
    model = EdgeCormorant(args.num_cg_levels, args.maxl, args.max_sh, args.num_channels, num_species,
                          args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                          args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                          charge_scale, args.gaussian_mask,
                          # args.top, args.input, args.num_mpnn_levels, args.num_top_levels, activation=args.top_activation,
                          top, input_type, mpnn_levels, num_top_levels, activation=top_activation,
                          additional_atom_features=additional_atom_features, 
                          # num_scalars_in=16,
                          device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype, remove_nonzero=True)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()


if __name__ == '__main__':
    main()
