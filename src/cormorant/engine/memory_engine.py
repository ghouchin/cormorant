import logging
import torch
import os
from datetime import datetime
from math import sqrt, inf, ceil
from cormorant.engine.utils import rel_pos_deriv_to_forces
from pytorch_memlab import profile
from pytorch_memlab import MemReporter 

MAE = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
RMSE = lambda x, y: sqrt(MSE(x, y))

logger = logging.getLogger(__name__)


class Engine(object):
    """
    Class for both training and inference phasees of the Cormorant network.

    Includes checkpoints, optimizer, scheduler.

    Roughly based upon TorchNet
    """

    def __init__(self, args, dataloaders, model, loss_fn, optimizer, scheduler, target, restart_epochs,
                 bestfile, checkfile, num_epoch, num_train, batch_size, device, dtype, save=True, load=True, alpha=0,
                 lr_minibatch=False, predictfile=None, textlog=True):
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.restart_epochs = restart_epochs
        self.lr_minibatch = lr_minibatch
        self.num_train = num_train
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.target = target
        self.alpha = alpha
        self.predictfile = predictfile
        self.textlog = textlog
        self.save = save
        self.load = load
        self.loss_fn = loss_fn

        self.stats = dataloaders['train'].dataset.stats
        self.bestfile = bestfile
        self.checkfile = checkfile

        # TODO: Fix this until TB summarize is implemented.
        self.summarize = False

        self.best_loss = inf
        self.epoch = 0
        self.minibatch = 0

        self.device = device
        self.dtype = dtype

    def _save_checkpoint(self, valid_loss):
        if not self.save:
            return

        save_dict = {'args': self.args,
                     'model_state': self.model.state_dict(),
                     'optimizer_state': self.optimizer.state_dict(),
                     'scheduler_state': self.scheduler.state_dict(),
                     'epoch': self.epoch,
                     'minibatch': self.minibatch,
                     'best_loss': self.best_loss}

        if (valid_loss < self.best_loss):
            self.best_loss = save_dict['best_loss'] = valid_loss
            logging.info('Lowest loss achieved! Saving best result to file: {}'.format(self.bestfile))
            torch.save(save_dict, self.bestfile)

        logging.info('Saving to checkpoint file: {}'.format(self.checkfile))
        torch.save(save_dict, self.checkfile)

    def load_checkpoint(self):
        """
        Load checkpoint from previous file.
        """
        if not self.load:
            return
        elif os.path.exists(self.checkfile):
            logging.info('Loading previous model from checkpoint!')
            self.load_state(self.checkfile)
        else:
            logging.info('No checkpoint included! Starting fresh training program.')
            return

    def load_state(self, checkfile):
        logging.info('Loading from checkpoint!')

        checkpoint = torch.load(checkfile)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.minibatch = checkpoint['minibatch']

        logging.info('Best loss from checkpoint: {} at epoch {}'.format(self.best_loss, self.epoch))

    def evaluate(self, splits=None, best=True, final=True):
        """
        Evaluate model on training/validation/testing splits.

        :splits: List of splits to include. Only valid splits are: 'train', 'valid', 'test'
        :best: Evaluate best model as determined by minimum validation error over evolution
        :final: Evaluate final model at end of training phase
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
        if not self.save:
            logging.info('No model saved! Cannot give final status.')
            return

        # Evaluate final model (at end of training)
        if final:
            logging.info('Getting predictions for model in last checkpoint.')

            # Load checkpoint model to make predictions
            checkpoint = torch.load(self.checkfile)
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets, losses = self.predict(split)
                self.log_predict(predict, targets, losses, split, description='Final')

        # Evaluate best model as determined by validation error
        if best:
            logging.info('Getting predictions for best model.')

            # Load best model to make predictions
            checkpoint = torch.load(self.bestfile)
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets, loss = self.predict(split)
                self.log_predict(predict, targets, loss, split, description='Best')
        logging.info('Inference phase complete!')

    def _warm_restart(self, epoch):
        restart_epochs = self.restart_epochs

        if epoch in restart_epochs:
            logging.info('Warm learning rate restart at epoch {}!'.format(epoch))
            self.scheduler.last_epoch = 0
            idx = restart_epochs.index(epoch)
            self.scheduler.T_max = restart_epochs[idx+1] - restart_epochs[idx]
            if self.lr_minibatch:
                self.scheduler.T_max *= ceil(self.num_train / self.batch_size)
            self.scheduler.step(0)

    def _log_minibatch(self, batch_idx, loss, mini_batch_mae, mini_batch_rmse, batch_t, epoch_t):
        mini_batch_loss = loss.item()
        #mini_batch_mae = MAE(predict, targets)
        #mini_batch_rmse = RMSE(predict, targets)

        # Exponential average of recent MAE/RMSE on training set for more convenient logging.
        if batch_idx == 0:
            self.mae, self.rmse = mini_batch_mae, mini_batch_rmse
        else:
            alpha = self.alpha
            self.mae = alpha * self.mae + (1 - alpha) * mini_batch_mae
            self.rmse = alpha * self.rmse + (1 - alpha) * mini_batch_rmse

        dtb = (datetime.now() - batch_t).total_seconds()
        tepoch = (datetime.now() - epoch_t).total_seconds()
        self.batch_time += dtb
        tcollate = tepoch-self.batch_time

        if self.textlog:
            logstring = ' E:{:3}, B: {:5}/{}'.format(self.epoch+1, batch_idx, len(self.dataloaders['train']))
            logstring += '{:> 9.4f}{:> 9.4f}{:> 9.4f}'.format(sqrt(mini_batch_loss), self.mae, self.rmse)
            logstring += '  dt:{:> 6.2f}{:> 8.2f}{:> 8.2f}'.format(dtb, tepoch, tcollate)
            logstring += '  {:.2E}'.format(self.scheduler.get_lr()[0])
            logging.info(logstring)

        if self.summarize:
            self.summarize.add_scalar('train/mae', sqrt(mini_batch_loss), self.minibatch)

    def _step_lr_batch(self):
        if self.lr_minibatch:
            self.scheduler.step()

    def _step_lr_epoch(self):
        if not self.lr_minibatch:
            self.scheduler.step()

    def train(self):
        epoch0 = self.epoch
        for epoch in range(epoch0, self.num_epoch):
            self.epoch = epoch
            # epoch_time = datetime.now()
            logging.info('Starting Epoch: {}'.format(epoch+1))

            self._warm_restart(epoch)
            self._step_lr_epoch()

            train_predict, train_targets, train_loss  = self.train_epoch()
            valid_loss, valid_mae, valid_rmse = self.predict('valid')
            #train_predict, train_targets, train_loss  = self.train_epoch()
            #valid_predict, valid_targets, valid_loss = self.predict('valid')

            self.log_predict(train_loss, self.mae, self.rmse, 'train', epoch=epoch)
            self.log_predict(valid_loss, vaild_mae, valid_rmse, 'valid', epoch=epoch)
            #train_loss, train_mae, train_rmse = self.log_predict(train_predict, train_targets, train_loss, 'train', epoch=epoch)
            #valid_loss, valid_mae, valid_rmse = self.log_predict(valid_predict, valid_targets, valid_loss, 'valid', epoch=epoch)


            #This needs to be changed so that valid_loss is passed to log_predict and not returned from it!
            self._save_checkpoint(valid_loss)

            logging.info('Epoch {} complete!'.format(epoch+1))

    def _get_target(self, data, stats=None):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """
        targets = data[self.target].to(self.device, self.dtype)
        

        if stats is not None:
            mu, sigma = stats[self.target]
            targets = (targets - mu) / sigma

        return targets/data['num_atoms'].to(self.device)

    #@profile
    def train_epoch(self):
        dataloader = self.dataloaders['train']

        self.mae, self.rmse, self.batch_time = 0, 0, 0
        all_loss = 0
        all_predict, all_targets = [], []
    

        self.model.train()
        epoch_t = datetime.now()
        for batch_idx, data in enumerate(dataloader):
            batch_t = datetime.now()

            # Calculate loss and backprop
            #import pdb
            #pdb.set_trace()
            loss, mse, mae = self.compute_single_batch(data)
            loss.backward()

            # ####### DEBUG
            # params = list(self.model.parameters())
            # print(torch.tensor([torch.max(p0.grad) for p0 in params]))
            # print(torch.tensor([torch.min(p0.grad) for p0 in params]))
            # print(torch.tensor([torch.min(torch.abs(p0.grad)) for p0 in params]))
            # raise Exception
            # ####### DEBUG

            # Step optimizer and learning rate
            self.optimizer.step()
            self._step_lr_batch()

            mse, mae  = mse.detach().cpu(), mae.detach().cpu()
            targets, predict = targets.detach().cpu(), predict.detach().cpu()
            loss = loss.detach().cpu()
            
            all_predict.append(predict)
            all_targets.append(targets)

            #all_mse += mse*data['energy'].shape[0] #generating sum of square errors
            #all_mae += mae*data['energy'].shape[0] #generating sum of abs errors
            all_loss += loss #*data['energy'].shape[0]

            self._log_minibatch(batch_idx, loss, mae, sqrt(mse), batch_t, epoch_t)

            self.minibatch += 1
        #N = len(dataloader.dataset.data['energy'])
        #all_rmse = sqrt(all_mse/N)
        #all_mae /= N
        #all_loss /= N
        #all_predict = torch.cat(all_predict)
        #all_targets = torch.cat(all_targets)


        return all_loss 

    def compute_single_batch(self, data):
        # Standard zero-gradient
        self.optimizer.zero_grad()

        targets = self._get_target(data, self.stats)
        predict = self.model(data)

        loss = self.loss_fn(predict, targets)
        return loss, predict, targets

    def predict(self, set='valid'):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_loss, all_mae, all_mse = 0, 0, 0
        start_time = datetime.now()
        logging.info('Starting testing on {} set: '.format(set))


        # for batch_idx, data in enumerate(dataloader):
        for data in dataloader:
            
            loss, mse, mae, predict, targets = self.compute_single_batch(data)

            all_mse += mse*data['energy'].shape[0] #generating sum of square errors
            all_mae += mae*data['energy'].shape[0] #generating sum of abs errors 
            all_loss += loss*data['energy'].shape[0]
        
        N = len(dataloader.dataset.data['energy'])
        all_rmse = sqrt(all_mse/N)
        all_mae /= N 

        dt = (datetime.now() - start_time).total_seconds()
        logging.info(' Done! (Time: {}s)'.format(dt))

        return all_loss, all_mae, all_rmse, predict, targets

    def log_predict(self, loss, mae, rmse, dataset, epoch=-1, description='Current'):
        #predict = predict.cpu().double()
        #targets = targets.cpu().double()

        #mae = MAE(predict, targets)
        #rmse = RMSE(predict, targets)

        mu, sigma = self.stats[self.target]
        mae_units = sigma*mae
        rmse_units = sigma*rmse
        #loss = torch.mean(losses)
        loss_units = sigma*loss

        datastrings = {'train': 'Training', 'test': 'Testing', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
            logstr = 'Epoch: {} Complete! {} {} Loss: {:8.4f} w/units: {:8.4f}.  MAE: {:8.4f} w/units: {:8.4f} RMSE: {:8.4f} w/units: {:8.4f}'
            logstr = logstr.format(epoch+1, description, datastrings[dataset], loss, loss_units, mae, mae_units, rmse, rmse_units)
            logging.info(logstr)
        else:
            suffix = 'best'
            logstr = 'Training: {} Complete! {} {} Loss: {:8.4f} w/units: {:8.4f}.  MAE: {:8.4f} w/units: {:8.4f} RMSE: {:8.4f} w/units: {:8.4f}'
            logstr = logstr.format(epoch+1, description, datastrings[dataset], loss, loss_units, mae, mae_units, rmse, rmse_units)
            logging.info(logstr)

        if self.predictfile is not None:
            file = self.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logging.info('Saving predictions to file: {}'.format(file))
            torch.save({'predict': predict, 'targets': targets}, file)

        #return loss, mae, rmse


class ForceEngine(Engine):
    def __init__(self, args, dataloaders, model, loss_fn, optimizer, scheduler, target, restart_epochs,
                 bestfile, checkfile, num_epoch, num_train, batch_size, device, dtype,
                 uses_relative_pos=False, save=True, load=True, alpha=0, lr_minibatch=False, predictfile=None, textlog=True):
        super().__init__(args, dataloaders, model, loss_fn, optimizer, scheduler, target, restart_epochs,
                         bestfile, checkfile, num_epoch, num_train, batch_size, device, dtype, save=save, load=load,
                         alpha=alpha, lr_minibatch=lr_minibatch, predictfile=predictfile, textlog=textlog)
        self.uses_relative_pos = uses_relative_pos

    #@profile
    def compute_single_batch(self, data):
        # Standard zero-gradient
        self.optimizer.zero_grad()

        # Get targets and predictions
        energy_scaled = self._get_target(data, self.stats)
        force_scaled = data['forces'].to(self.device, self.dtype)
        if self.stats is not None:
            __, sigma = self.stats[self.target]
            force_scaled /= sigma
        s = force_scaled.shape

        force_scaled = (force_scaled.view(s[0],-1).t()/data['num_atoms'].to(self.device)).t().view(s) #hack to get force on an atom divided by num_atoms
        
        data['relative_pos'] = data['relative_pos'].to(self.device, self.dtype)
        data['positions'] = data['positions'].to(self.device, self.dtype)

        if self.uses_relative_pos:
            pos_input = data['relative_pos']
        else:
            pos_input = data['positions']
        pos_input.requires_grad_()

        energy_pred = self.model(data)

        Esum = torch.sum(energy_pred, dim=0)
        force_pred = -torch.autograd.grad(Esum, pos_input, create_graph=True, retain_graph=True)[0]
        if self.uses_relative_pos:
            force_pred[~data['edge_mask']] = 0.
            force_pred = rel_pos_deriv_to_forces(force_pred)
        else:
            force_pred[~data['atom_mask']] = 0.

        force_pred = (force_pred.view(s[0],-1).t()/data['num_atoms'].to(self.device)).t().view(s)

        # Calculate loss and backprop
        loss, mse = self.loss_fn(energy_pred/data['num_atoms'].to(self.device), force_pred, energy_scaled, force_scaled)
        mae = torch.nn.functional.l1_loss(energy_pred, energy_scaled)
        return loss, mse, mae
        


def energy_and_force_mse_loss(energy_pred, force_pred, energy_scaled, force_scaled, force_factor=0.):
    """
    Basic MSE loss on the energies and forces
    """
    #import pdb
    #pdb.set_trace()
    energy_mse = torch.nn.functional.mse_loss(energy_pred, energy_scaled)
    energy_mae = torch.nn.functional.l1_loss(energy_pred, energy_scaled)
    force_mse = torch.nn.functional.mse_loss(force_pred, force_scaled)
    # loss = energy_mse + force_factor * force_mse
    loss = energy_mse  + force_factor * force_mse
    return loss, energy_mse
