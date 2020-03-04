import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import os
import sys
from datetime import datetime
from math import log, log2, exp, ceil, sqrt

import logging
logger = logging.getLogger(__name__)

# if sys.version_info < (3, 6):
#     logger.info('Cormorant requires Python version 3.6! or above!')
#     sys.exit(1)

MAE = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
RMSE = lambda x, y: sqrt(MSE(x, y))

# ### Initialize parameters for training run ####


def init_argparse(dataset):
    """
    Reads in the arguments for the script for a given dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.  Currently 'md17' and 'qm9' are supported.

    Returns
    -------
    args : :class:`Namespace`
        Namespace with a dictionary of arguments where the key is the name of
        the argument and the item is the input value.
    """
    from cormorant.engine.args import setup_argparse

    parser = setup_argparse(dataset)
    args = parser.parse_args()
    d = vars(args)
    d['dataset'] = dataset

    return args


def init_logger(logfile, log_level):
    if logfile:
        handlers = [logging.FileHandler(logfile, mode='w'), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    if log_level.lower() == 'debug':
        loglevel = logging.DEBUG
    elif log_level.lower() == 'info':
        loglevel = logging.INFO
    else:
        ValueError('Inappropriate choice of logging_level. {}'.format(log_level))

    logging.basicConfig(level=loglevel,
                        format='%(message)s',
                        handlers=handlers
                        )


# def init_file_paths(dataset, prefix, workdir, modeldir, logdir, predictdir, logfile=None, bestfile=None, checkfile=None, loadfile=None, predictfile=None):
#     # Initialize files and directories to load/save logs, models, and predictions
#     workdir = args.workdir
#     prefix = args.prefix
#     modeldir = args.modeldir
#     logdir = args.logdir
#     predictdir = args.predictdir
#
#     if prefix and not args.logfile:
#         args.logfile = os.path.join(workdir, logdir, prefix+'.log')
#     if prefix and not args.bestfile:
#         args.bestfile = os.path.join(workdir, modeldir, prefix+'_best.pt')
#     if prefix and not args.checkfile:
#         args.checkfile = os.path.join(workdir, modeldir, prefix+'.pt')
#     if prefix and not args.loadfile:
#         args.loadfile = args.checkfile
#     if prefix and not args.predictfile:
#         args.predictfile = os.path.join(workdir, predictdir, prefix)
#
#     if not os.path.exists(modeldir):
#         logger.warning('Model directory {} does not exist. Creating!'.format(modeldir))
#         os.mkdir(modeldir)
#     if not os.path.exists(logdir):
#         logger.warning('Logging directory {} does not exist. Creating!'.format(logdir))
#         os.mkdir(logdir)
#     if not os.path.exists(predictdir):
#         logger.warning('Prediction directory {} does not exist. Creating!'.format(predictdir))
#         os.mkdir(predictdir)
#
#     args.dataset = args.dataset.lower()
#
#     if args.dataset.startswith('qm9'):
#         if not args.target:
#             args.target = 'U0'
#     elif args.dataset.startswith('md17'):
#         if not args.subset:
#             args.subset = 'uracil'
#         if not args.target:
#             args.target = 'energies'
#     elif args.dataset.startswith('ase-db'):
#         if not args.target:
#             args.target = 'energy'
#     else:
#         raise ValueError('Dataset must be qm9 or md17 or an ASE Database!')
#
#     return args


def init_file_paths(prefix, workdir, modeldir, logdir, predictdir, logfile=None, bestfile=None, checkfile=None, loadfile=None, predictfile=None):
    # Initialize files and directories to load/save logs, models, and predictions
    if prefix and not logfile:
        logfile = os.path.join(workdir, logdir, prefix+'.log')
    if prefix and not bestfile:
        bestfile = os.path.join(workdir, modeldir, prefix+'_best.pt')
    if prefix and not checkfile:
        checkfile = os.path.join(workdir, modeldir, prefix+'.pt')
    if prefix and not loadfile:
        loadfile = checkfile
    if prefix and not predictfile:
        predictfile = os.path.join(workdir, predictdir, prefix)

    if not os.path.exists(modeldir):
        logger.warning('Model directory {} does not exist. Creating!'.format(modeldir))
        os.mkdir(modeldir)
    if not os.path.exists(logdir):
        logger.warning('Logging directory {} does not exist. Creating!'.format(logdir))
        os.mkdir(logdir)
    if not os.path.exists(predictdir):
        logger.warning('Prediction directory {} does not exist. Creating!'.format(predictdir))
        os.mkdir(predictdir)
    # return prefix, workdir, modeldir, logdir, predictdir
    return logfile, bestfile, checkfile, loadfile, predictfile


def set_dataset_defaults(args):
    args.dataset = args.dataset.lower()
    if args.dataset.startswith('qm9'):
        if not args.target:
            args.target = 'U0'
    elif args.dataset.startswith('md17'):
        if not args.subset:
            args.subset = 'uracil'
        if not args.target:
            args.target = 'energies'
    elif args.dataset.startswith('ase-db'):
        if not args.target:
            args.target = 'energy'
    else:
        raise ValueError('Dataset must be qm9 or md17 or an ASE Database!')
    return args


def logging_printout(args):

    # Printouts of various inputs before training (and after logger is initialized with correct logfile path)
    logger.info('Initializing simulation based upon argument string:')
    logger.info(' '.join([arg for arg in sys.argv]))
    logger.info('Log, best, checkpoint, load files: {} {} {} {}'.format(args.logfile, args.bestfile, args.checkfile, args.loadfile))
    logger.info('Dataset, learning target, datadir: {} {} {}'.format(args.dataset, args.target, args.datadir))

    if args.seed < 0:
        seed = int((datetime.now().timestamp())*100000)
        logger.info('Setting seed based upon time: {}'.format(seed))
        args.seed = seed
        torch.manual_seed(seed)

    logger.info('Values of all model arguments:')
    logger.info('{}'.format(args))


def init_optimizer(model, optim_type, lr_init, weight_decay):
    """
    Convenience function for initializing the optimizer.

    Parameters
    ----------
    model : pytorch module
        model to be trained
    optim : string
        Optimizer to use.
    lr_init : float
        Initial learning rate
    weight_decay :  ???
        Weight decay

    Returns
    -------
    optimizer : pytorch optimizer
        Optimizer for the model
    """

    params = {'params': model.parameters(), 'lr': lr_init, 'weight_decay': weight_decay}
    params = [params]

    optim_type = optim_type.lower()

    if optim_type == 'adam':
        optimizer = optim.AdamW(params, amsgrad=False)
    elif optim_type == 'amsgrad':
        optimizer = optim.Adam(params, amsgrad=True)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(params)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(params)
    else:
        raise ValueError('Incorrect choice of optimizer')

    return optimizer


def init_scheduler(optimizer, lr_init, lr_final, lr_decay, num_epoch, num_train, batch_size, sgd_restart=0, lr_minibatch=False, lr_decay_type='cos'):
    lr_decay = min(lr_decay, num_epoch)

    minibatch_per_epoch = ceil(num_train / batch_size)
    if lr_minibatch:
        lr_decay = lr_decay*minibatch_per_epoch

    lr_ratio = lr_final/lr_init

    lr_bounds = lambda lr, lr_min: min(1, max(lr_min, lr))
    if sgd_restart > 0:
        restart_epochs = [(2**k-1) for k in range(1, ceil(log2(num_epoch))+1)]
        lr_hold = restart_epochs[0]
        if lr_minibatch:
            lr_hold *= minibatch_per_epoch
        logger.info('SGD Restart epochs: {}'.format(restart_epochs))
    else:
        restart_epochs = []
        lr_hold = num_epoch
        if lr_minibatch:
            lr_hold *= minibatch_per_epoch

    if lr_decay_type.startswith('cos'):
        scheduler = sched.CosineAnnealingLR(optimizer, lr_hold, eta_min=lr_final)
    elif lr_decay_type.startswith('exp'):
        lr_lambda = lambda epoch: lr_bounds(exp(epoch / lr_decay) * log(lr_ratio), lr_ratio)
        scheduler = sched.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError('Incorrect choice for lr_decay_type!')

    return scheduler, restart_epochs


def init_cuda(cuda, dtype):
    if cuda:
        assert(torch.cuda.is_available()), 'No CUDA device available!'
        logger.info('Beginning training on CUDA/GPU! Device: {}'.format(torch.cuda.current_device()))
        torch.cuda.init()
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        logger.info('Beginning training on CPU!')
        device = torch.device('cpu')

    if dtype == 'double':
        dtype = torch.double
    elif dtype == 'float':
        dtype = torch.float
    else:
        raise ValueError('Incorrect data type chosen!')

    return device, dtype
