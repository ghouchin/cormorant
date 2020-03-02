import torch
from torch.utils.data import DataLoader
from cormorant.models import CormorantMD17
from cormorant.models.autotest import cormorant_tests
from cormorant.engine import Engine
from cormorant.engine import init_file_paths, init_logger, logging_printout, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets
from cormorant.data.collate import collate_fn


def main():
    saved_run = torch.load("model/nosave_best.pt")
    args = saved_run['args']

    # Initialize device and data type
    device, dtype = init_cuda(args.cuda, args.dtype)

    # Initialize dataloader
    datadir = "/home/erik/My_Source_Codes/cormorant/examples/data/"
    ntr, nv, nte, datasets, num_species, charge_scale = initialize_datasets(args.num_train, args.num_valid, args.num_test,
                                                                       datadir, 'md17', subtract_thermo=False,
                                                                       force_download=args.force_download,
                                                                       subset='uracil'
                                                                       )
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

    model.load_state_dict(saved_run['model_state'])

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(model, args.optim, args.lr_init, args.weight_decay)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                   for split, dataset in datasets.items()}
    
    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)
    sample_batch = next(iter(dataloaders['train']))
    predict = model(sample_batch) 
    print(predict)



if __name__ == '__main__':
    main()
