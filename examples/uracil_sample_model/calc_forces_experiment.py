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

    optimizer.zero_grad()

    sample_batch['positions'].requires_grad_() # Need gradients on the positions
    
    print(sample_batch.keys())
    print(sample_batch['positions'].shape, "positions")
    print(sample_batch['charges'].shape, "charges")
    print(sample_batch['one_hot'].shape, "one_hot")
    print(sample_batch['atom_mask'].shape, "atom_mask")
    print(sample_batch['atom_mask'].type(), "atom_mask")
    print(sample_batch['atom_mask'].bool().type(), "atom_mask")

    predict = model(sample_batch)
    forces = []
    # Grad must be called for each predicted energy in the batch
    for i, pred in enumerate(predict):
        chunk_forces = -torch.autograd.grad(pred, sample_batch['positions'], create_graph=True, retain_graph=True)[0]
        forces.append(chunk_forces[i])
    forces = torch.stack(forces, dim=0)

    # Normalize forces by stdev of energy
    normed_forces = sample_batch['forces'] / datasets['train'].stats['energies'][1]
    
    print("Ratio of predicted to true force for configuration 0")
    print(forces[0] / normed_forces[0])
    print("Ratio of predicted to true force for configuration 1")
    print(forces[1] / normed_forces[1])
    print('calculate difference and optimize')
    loss_fn = torch.nn.functional.mse_loss(forces, normed_forces)
    loss_fn.backward()
    optimizer.step()







if __name__ == '__main__':
    main()
