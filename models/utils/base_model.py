# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ast
import datetime
from abc import abstractmethod
from typing import Union

import torch.nn as nn
import torch
from argparse import Namespace
from utils.conf import get_device
from datasets import BaseDataset
from utils.wandbsc import innested_vars
import os
from torch.optim.lr_scheduler import MultiStepLR


class BaseModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, dataset: BaseDataset) -> None:
        super(BaseModel, self).__init__()

        self.net = backbone

        self.loss = loss
        self.args = args
        self.transform = dataset.get_transform()
        self.device = get_device()
        setattr(self, f'best_{self.args.checkpoint_metric}', torch.tensor(float('inf')))
        self.ade_val = torch.tensor(float('inf'))
        self.fde_val = torch.tensor(float('inf'))
        self.loss_val = torch.tensor(float('inf'))
        self.ade_tf_val = torch.tensor(float('inf'))
        self.fde_tf_val = torch.tensor(float('inf'))
        self.loss_tf_val = torch.tensor(float('inf'))
        assert self.args.checkpoint_metric in ['loss_val', 'ade_val', 'fde_val', 'loss_tf_val', 'ade_tf_val',
                                               'fde_tf_val'], 'checkpoint_metric must be one of loss_val, ade_val, ' \
                                                              'fde_val, loss_tf_val, ade_tf_val, fde_tf_val'
        # load checkpoint

        self.optimizer = getattr(torch.optim, self.args.optimizer)
        self.optim_kwargs = {x.replace('opt_', ''): ast.literal_eval(y) for x, y in vars(self.args).items() if
                             x.startswith('opt_') and y is not None}
        # redo a literal eval to avoid problems with double quotes
        for key, value in self.optim_kwargs.items():
            if isinstance(value, str):
                try:
                    self.optim_kwargs[key] = ast.literal_eval(value)
                except:
                    pass
        self.opt = self.optimizer(self.net.parameters(), lr=self.args.lr,
                                  **self.optim_kwargs)

        self.scheduler = None

        train_loader, _, _, _ = dataset.get_data_loaders()

        self.dataset_name = dataset.NAME

        self.log_results = []
        self.wb_log = {}
        self.epoch = 0
        self.scheduler = self.set_scheduler()

        if args.checkpoint is not None:
            self.load_checkpoint_states_dict(args.checkpoint)
        self.step = 0
        self.callbacks = []
        self.total_steps = self.args.n_epochs * len(train_loader)

    def get_name(self):
        return self.NAME.capitalize()

    def load_checkpoint_states_dict(self, ckp, *args, **kwargs):
        self.net.load_state_dict(ckp['model_state_dict'])
        self.opt.load_state_dict(ckp['optimizer_state_dict'])
        self.scheduler.load_state_dict(
            ckp['lr_scheduler_state_dict']) if self.scheduler is not None else None

    def forward(self, x: torch.Tensor, tgt: Union[torch.Tensor, None], seq_start_end: torch.Tensor) -> dict:
        """
        Computes a forward pass.
        @param seq_start_end:
        @param x:
        @param tgt:
        :return: the result of the computation
        """
        if tgt is None:
            output = {'pred': self.net(x)}
        else:
            output = {'pred': self.net(x), 'loss': self.loss(self.net(x), tgt)}
        return output

    @abstractmethod
    def observe(self, src: torch.Tensor, tgt: torch.Tensor, seq_start_end: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param data: batch of examples
        :return: the value of the loss function
        @param seq_start_end:
        @param tgt:
        @param src:
        """
        pass

    def log_accs(self, accs):
        pass

    def save_logs(self):
        log_dir = os.path.join(self.result_dir, 'logs', self.dataset_name, self.args.name)

        obj = {**innested_vars(self.args), 'results': self.log_results}
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filename = f'{self.wblogger.run_id}.pyd'
        with open(os.path.join(log_dir, filename), 'a') as f:
            f.write(str(obj) + '\n')
        return log_dir

    def set_scheduler(self, optimizer=None):
        if optimizer is None:
            optimizer = self.opt
        if self.args.scheduler is not None and self.args.n_epochs > 1:
            if self.args.scheduler == 'multistep':
                scheduler = MultiStepLR(optimizer, milestones=self.args.lr_decay_steps, gamma=self.args.lr_decay)
            else:
                raise ValueError(f'Unknown scheduler {self.args.scheduler}')
            return scheduler
        else:
            return None

    def on_epoch_start(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(self, dataset, *args, **kwargs)

    @torch.no_grad()
    def evaluate(self, loader, *args, **kwargs):
        pass

    def scheduler_step(self, *args, **kwargs):
        # eventually scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(self, name='best', metric=None, epoch=None):
        if self.args.save_checks:
            ckp_name = f'checkpoint_{name}.pth'
            if metric is None:
                metric = getattr(self, f'best_{self.args.checkpoint_metric}')
            lr_scheduler_state_dict = None
            if self.scheduler is not None:
                lr_scheduler_state_dict = self.scheduler.state_dict()
            torch.save({'args': self.args.__dict__,
                        'epoch': self.epoch if epoch is None else epoch,
                        'best_ade': metric,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler_state_dict,
                        }, os.path.join(self.result_dir, ckp_name))

    def on_epoch_end(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(self, dataset, *args, **kwargs)
        if self.args.save_checks:
            if self.epoch % self.args.save_every == 0:
                # update best metrics
                actual_metric = getattr(self, self.args.checkpoint_metric)
                best_metric = getattr(self, f'best_{self.args.checkpoint_metric}')
                if actual_metric <= best_metric:
                    setattr(self, f'best_{self.args.checkpoint_metric}', actual_metric)
                    self.save_checkpoint('best')
                self.save_checkpoint('last', epoch=self.epoch, metric=actual_metric)

    def on_training_end(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end(self, dataset, *args, **kwargs)

    def on_training_start(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start(self, dataset, *args, **kwargs)

    def after_training_epoch(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'after_training_epoch'):
                callback.after_training_epoch(self, dataset, *args, **kwargs)

    def on_batch_start(self, dataset, *args, **kwargs):
        # running all the callbacks that are registered for the batch start
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_start'):
                callback.on_batch_start(self, dataset, *args, **kwargs)

    def on_batch_end(self, dataset, *args, **kwargs):
        # running all the callbacks that are registered for the batch start
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(self, dataset, *args, **kwargs)

    def on_test_start(self, dataset, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_test_start'):
                callback.on_test_start(self, dataset, *args, **kwargs)

    def update_checkpoint(self):
        metric = getattr(self, f'best_{self.args.checkpoint_metric}')
        if os.path.isfile(os.path.join(self.result_dir, 'tested.pth')):
            start_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_'
        else:
            start_name = '_'
        lr_scheduler_state_dict = None
        if self.scheduler is not None:
            lr_scheduler_state_dict = self.scheduler.state_dict()
        torch.save({'args': self.args.__dict__,
                    'epoch': self.epoch,
                    'best_ade': metric,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler_state_dict,
                    }, os.path.join(self.result_dir, start_name + 'tested.pth'))
