import logging
import torch
import os
from datetime import datetime
from math import sqrt, inf, ceil
from cormorant.engine.utils import rel_pos_deriv_to_forces

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
                     'best_loss': self.best_loss,
                     'stats': self.stats,
                     'max_charge': self.dataloaders['train'].dataset.max_charge,
                     'included_species': self.dataloaders['train'].dataset.included_species,
                     'num_species': self.dataloaders['train'].dataset.num_species}

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
                losses, mae, rmse, predict, targets = self.predict(split)
                self.log_predict(predict, targets, losses, mae, rmse, split, description='Final')

        # Evaluate best model as determined by validation error
        if best:
            logging.info('Getting predictions for best model.')

            # Load best model to make predictions
            checkpoint = torch.load(self.bestfile)
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                loss, mae, rmse, predict, targets = self.predict(split)
                self.log_predict(predict, targets, loss, mae, rmse, split, description='Best')
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
            valid_loss, valid_mae, valid_rmse, valid_predict, valid_targets = self.predict('valid')

            self.log_predict(train_predict, train_targets, train_loss, self.mae, self.rmse, 'train', epoch=epoch)
            self.log_predict(valid_predict, valid_targets, valid_loss, valid_mae, valid_rmse, 'valid', epoch=epoch)

            self._save_checkpoint(valid_loss)

            logging.info('Epoch {} complete!'.format(epoch+1))

    def _get_target(self, data, stats=None):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """
        targets = data[self.target].to(self.device, self.dtype)
        

        if stats is not None:
            mu, sigma = stats[self.target] #mu is the average energy per atom and std is the standard deviation of the total energies.
            targets = (targets - mu*data['num_atoms'].to(self.device)) / sigma

        return targets

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
            loss, mse, mae, predict, targets = self.compute_single_batch(data)
            mae = mae.detach().cpu()
            loss.backward()


            # Step optimizer and learning rate
            self.optimizer.step()
            self._step_lr_batch()

            mse = mse.detach().cpu() 
            targets, predict = targets.detach().cpu(), predict.detach().cpu()
            loss = loss.detach().cpu()
            
            all_predict.append(predict)
            all_targets.append(targets)

            all_loss += loss 

            #the running total of MAE and RMSE is calcuated here and saved to self.mae, self.rmse
            self._log_minibatch(batch_idx, loss, mae, sqrt(mse), batch_t, epoch_t)

            self.minibatch += 1

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets, all_loss 

    def compute_single_batch(self, data):
        # Standard zero-gradient
        self.optimizer.zero_grad()
        targets = self._get_target(data, self.stats)
        predict = self.model(data)
        data['num_atoms'] = data['num_atoms'].to(self.device)
        loss = self.loss_fn(predict/data['num_atoms'], targets/data['num_atoms'])
        mse = loss
        mae = torch.nn.functional.l1_loss(predict/data['num_atoms'], targets/data['num_atoms'])

        mu, sigma = self.stats[self.target] #mu is the average energy per atom and std is the standard deviation of the total energies. 
        predict = sigma*predict + mu*data['num_atoms']
        targets = sigma*targets + mu*data['num_atoms']
        predict = predict/data['num_atoms']
        targets = targets/data['num_atoms']

        data['num_atoms'] = data['num_atoms'].detach().cpu()
        return loss, mse, mae, predict, targets

    def predict(self, set='valid'):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_loss, all_mae, all_mse = 0, 0, 0
        all_predict, all_targets = [], []
        start_time = datetime.now()
        logging.info('Starting testing on {} set: '.format(set))


        for data in dataloader:
            
            loss, mse, mae, predict, targets = self.compute_single_batch(data)
            loss = loss.detach().cpu()
            mse = mse.detach().cpu()
            mae = mae.detach().cpu()

            predict, targets = targets.detach().cpu(), predict.detach().cpu()
            all_targets.append(targets)
            all_predict.append(predict)

            all_mse += mse*data['energy'].shape[0] #generating sum of square errors
            all_mae += mae*data['energy'].shape[0] #generating sum of abs errors 
            all_loss += loss*data['energy'].shape[0]
        
        N = len(dataloader.dataset.data['energy'])
        all_rmse = sqrt(all_mse/N)
        all_mae /= N 
        all_loss /= N
        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        dt = (datetime.now() - start_time).total_seconds()
        logging.info(' Done! (Time: {}s)'.format(dt))

        return all_loss, all_mae, all_rmse, all_predict, all_targets
        

    def log_predict(self, predict, targets, loss, mae, rmse, dataset, epoch=-1, description='Current'):
        mu, sigma = self.stats[self.target]
        mae_units = sigma*mae
        rmse_units = sigma*rmse
        loss_units = sigma*sigma*loss


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


class ForceEngine(Engine):
    def __init__(self, args, dataloaders, model, loss_fn, optimizer, scheduler, target, restart_epochs,
                 bestfile, checkfile, num_epoch, num_train, batch_size, device, dtype,
                 uses_relative_pos=False, save=True, load=True, alpha=0, lr_minibatch=False, predictfile=None, textlog=True):
        super().__init__(args, dataloaders, model, loss_fn, optimizer, scheduler, target, restart_epochs,
                         bestfile, checkfile, num_epoch, num_train, batch_size, device, dtype, save=save, load=load,
                         alpha=alpha, lr_minibatch=lr_minibatch, predictfile=predictfile, textlog=textlog)
        self.uses_relative_pos = uses_relative_pos

    def train(self):
        epoch0 = self.epoch
        for epoch in range(epoch0, self.num_epoch):
            self.epoch = epoch
            # epoch_time = datetime.now()                                                                                                                                                                                                                                                                    
            logging.info('Starting Epoch: {}'.format(epoch+1))

            self._warm_restart(epoch)
            self._step_lr_epoch()

            train_predict, train_targets, train_loss  = self.train_epoch()
            valid_loss, valid_mae, valid_rmse, valid_force_rmse, valid_predict, valid_targets = self.predict('valid')

            self.log_predict(train_predict, train_targets, train_loss, self.mae, self.rmse, self.force_rmse, 'train', epoch=epoch)
            self.log_predict(valid_predict, valid_targets, valid_loss, valid_mae, valid_rmse, valid_force_rmse, 'valid', epoch=epoch)

            self._save_checkpoint(valid_loss)

            logging.info('Epoch {} complete!'.format(epoch+1))


    def train_epoch(self):
        dataloader = self.dataloaders['train']

        self.mae, self.rmse, self.force_rmse, self.batch_time = 0, 0, 0, 0
        all_loss = 0
        all_predict, all_targets = [], []


        self.model.train()
        epoch_t = datetime.now()
        for batch_idx, data in enumerate(dataloader):
            batch_t = datetime.now()

            # Calculate loss and backprop                                                                                                                                                                                                                                                                    
            loss, mse, mae, force_mse, predict, targets = self.compute_single_batch(data)
            mae = mae.detach().cpu()
            loss.backward()

            # Step optimizer and learning rate                                                                                                                                                                                                                                                               
            self.optimizer.step()
            self._step_lr_batch()

            mse = mse.detach().cpu()
            force_mse = force_mse.detach().cpu()
            targets, predict = targets.detach().cpu(), predict.detach().cpu()
            loss = loss.detach().cpu()

            all_predict.append(predict)
            all_targets.append(targets)

            all_loss += loss

            #the running total of MAE and RMSE is calcuated here and saved to self.mae, self.rmse                                                                                                                                                                                                            
            self._log_minibatch(batch_idx, loss, mae, sqrt(mse), sqrt(force_mse), batch_t, epoch_t)
            

            self.minibatch += 1

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets, all_loss


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

        #force_scaled = (force_scaled.view(s[0],-1).t()/data['num_atoms'].to(self.device)).t().view(s) #hack to get force on an atom divided by num_atoms
        
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

        #force_pred = (force_pred.view(s[0],-1).t()/data['num_atoms'].to(self.device)).t().view(s)
        data['num_atoms'] = data['num_atoms'].to(self.device)
        # Calculate loss and backprop
        loss, mse, force_mse = self.loss_fn(energy_pred/data['num_atoms'], force_pred, energy_scaled/data['num_atoms'], force_scaled)
        mae = torch.nn.functional.l1_loss(energy_pred/data['num_atoms'], energy_scaled/data['num_atoms'])

        mu, sigma = self.stats[self.target]
        energy_pred = sigma*energy_pred + mu*data['num_atoms']
        energy_scaled = sigma*energy_scaled + mu*data['num_atoms']
        energy_pred = energy_pred/data['num_atoms']
        energy_scaled = energy_scaled/data['num_atoms']
        
        data['num_atoms'] = data['num_atoms'].detach().cpu()
        return loss, mse, mae, force_mse, energy_pred, energy_scaled  
        

    def predict(self, set='valid'):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_loss, all_mae, all_mse, all_force_mse = 0, 0, 0, 0
        all_predict, all_targets = [], []
        start_time = datetime.now()
        logging.info('Starting testing on {} set: '.format(set))


        for data in dataloader:

            loss, mse, mae, force_mse, predict, targets = self.compute_single_batch(data)
            loss = loss.detach().cpu()
            mse = mse.detach().cpu()
            mae = mae.detach().cpu()
            force_mse = force_mse.detach().cpu()
            predict, targets = targets.detach().cpu(), predict.detach().cpu()
            all_targets.append(targets)
            all_predict.append(predict)

            all_mse += mse*data['energy'].shape[0] #generating sum of square errors
            all_mae += mae*data['energy'].shape[0] #generating sum of abs errors
            all_force_mse += force_mse*data['energy'].shape[0] #generating sum of square errors
            all_loss += loss*data['energy'].shape[0]

        N = len(dataloader.dataset.data['energy'])
        all_rmse = sqrt(all_mse/N)
        all_force_rmse = sqrt(all_force_mse/N)
        all_mae /= N
        all_loss /= N
        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        dt = (datetime.now() - start_time).total_seconds()
        logging.info(' Done! (Time: {}s)'.format(dt))

        return all_loss, all_mae, all_rmse, all_force_rmse, all_predict, all_targets


    def _log_minibatch(self, batch_idx, loss, mini_batch_mae, mini_batch_rmse, mini_batch_force_rmse, batch_t, epoch_t):
        mini_batch_loss = loss.item()

        # Exponential average of recent MAE/RMSE on training set for more convenient logging.                                                                                                                                                                                                                
        if batch_idx == 0:
            self.mae, self.rmse, self.force_rmse = mini_batch_mae, mini_batch_rmse, mini_batch_force_rmse
        else:
            alpha = self.alpha
            self.mae = alpha * self.mae + (1 - alpha) * mini_batch_mae
            self.rmse = alpha * self.rmse + (1 - alpha) * mini_batch_rmse
            self.force_rmse = alpha * self.force_rmse + (1 - alpha) * mini_batch_force_rmse
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



    def log_predict(self, predict, targets, loss, mae, rmse, force_rmse, dataset, epoch=-1, description='Current'):
        mu, sigma = self.stats[self.target]
        mae_units = sigma*mae
        rmse_units = sigma*rmse
        loss_units = sigma*sigma*loss
        force_rmse_units = sigma*force_rmse


        datastrings = {'train': 'Training', 'test': 'Testing', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
            logstr = 'Epoch: {} Complete! {} {} Loss: {:8.4f} w/units: {:8.4f}.  MAE: {:8.4f} w/units: {:8.4f} RMSE: {:8.4f} w/units: {:8.4f}, FRMSE: {:8.4f} w/units: {:8.4f}'
            logstr = logstr.format(epoch+1, description, datastrings[dataset], loss, loss_units, mae, mae_units, rmse, rmse_units, force_rmse, force_rmse_units)
            logging.info(logstr)
        else:
            suffix = 'best'
            logstr = 'Training: {} Complete! {} {} Loss: {:8.4f} w/units: {:8.4f}.  MAE: {:8.4f} w/units: {:8.4f} RMSE: {:8.4f} w/units: {:8.4f}, FRMSE: {:8.4f} w/units: {:8.4f}'
            logstr = logstr.format(epoch+1, description, datastrings[dataset], loss, loss_units, mae, mae_units, rmse, rmse_units, force_rmse, force_rmse_units)
            logging.info(logstr)

        if self.predictfile is not None:
            file = self.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logging.info('Saving predictions to file: {}'.format(file))
            torch.save({'predict': predict, 'targets': targets}, file)



def energy_and_force_mse_loss(energy_pred, force_pred, energy_scaled, force_scaled, force_factor=0.):
    """
    Basic MSE loss on the energies and forces
    """
    energy_mse = torch.nn.functional.mse_loss(energy_pred, energy_scaled)
    force_mse = torch.nn.functional.mse_loss(force_pred, force_scaled)
    loss = energy_mse  + force_factor * force_mse
    return loss, energy_mse, force_mse
