# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
from models.utils.base_model import BaseModel

from utils.eval import evaluate
from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset
import sys

from utils.status import progress_bar


def train(model: BaseModel, dataset: BaseDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)

    start_epch = args.checkpoint['epoch'] if (args.checkpoint is not None) and (not args.reset_checkpoint_epoch) else 0
    args.n_epochs = 0 if args.only_eval else args.n_epochs
    epoch = start_epch
    print(f'Starting from epoch {start_epch} and training for {args.n_epochs} epochs', flush=True)
    print(file=sys.stderr)
    train_loader, val_loader, test_loader, train_sampler = dataset.get_data_loaders()
    # update model step
    model.step = start_epch * len(train_loader)
    model.epoch = start_epch
    # store the collate function to be used for training and testing
    collate_fun_train = train_loader.collate_fn
    collate_fun_test = test_loader.collate_fn
    model.on_training_start(dataset)
    for epoch in range(start_epch, args.n_epochs):
        model.train()
        model.on_epoch_start(dataset)
        for i, data in enumerate(train_loader):
            model.on_batch_start(dataset)
            data = [tensor.to(model.device) for tensor in data]
            src, tgt, seq_start_end = data
            loss = model.observe(src, tgt, seq_start_end)
            model.wblogger({'training': {'loss': loss, **model.wb_log}, 'epoch': epoch, 'batch': i,
                            'model_step': epoch * len(train_loader) + i})
            progress_bar(i, len(train_loader), epoch, loss, batch=f'{i + 1}/{len(train_loader)}', **model.wb_log)
            model.wb_log = {}
            model.on_batch_end(dataset)
            model.step += 1
        print('\nEpoch finished', flush=True)
        wb_logs = {'epoch': epoch}

        if len(model.opt.param_groups) > 1:
            wb_logs.update(
                {
                    f'learning_rate_{i}': gr['lr'] for i, gr in enumerate(
                    model.opt.param_groups)
                }
            )
        else:
            wb_logs.update({'learning_rate': model.opt.param_groups[0]['lr']})

        model.after_training_epoch(dataset)
        if not epoch % args.validate_every:
            # calculate teacher-forcing metrics for validation
            accs = evaluate(model, val_loader, epoch, teacher_forcing=True)
            for k, v in accs.items():
                setattr(model, k + '_val', v)
            # calculate teacher-forcing metrics for training
            train_loader.collate_fn = collate_fun_test
            accs_training = evaluate(model, train_loader, epoch, teacher_forcing=True)
            train_loader.collate_fn = collate_fun_train

            accs_training = {f'training_eval.{k}': v for k, v in accs_training.items()}

            print(f'\nEpoch {epoch}: ' + ' '.join(
                [f'Validation: {k}: {v:.4f}' for k, v in accs.items() if 'fig' not in k]))
            print(f'\nEpoch {epoch}: ' + ' '.join(
                [f'Training: {k}: {v:.4f}' for k, v in accs_training.items() if 'fig' not in k]))
            wb_logs.update({'validation': accs})
            wb_logs.update(accs_training)
            model.wblogger(wb_logs)

        model.scheduler_step()
        model.on_epoch_end(dataset)
        model.epoch += 1
        print(wb_logs)
    model.on_training_end(dataset)

    if args.save_checks and not args.only_eval:
        checkpoint = torch.load(os.path.join(model.result_dir, 'checkpoint_best.pth'), map_location='cpu')
        model.net.load_state_dict(checkpoint['model_state_dict'])

    model.on_test_start(dataset)
    accs = evaluate(model, test_loader, epoch, args.validate_with_tf)
    if args.save_checks:
        model.update_checkpoint()
    print(f'\nEpoch {epoch}: ' + ' '.join([f'Test: {k}: {v:.4f}' for k, v in accs.items() if 'fig' not in k]))
    if hasattr(model, 'log_accs'):
        model.log_accs(accs)
    log_obj = {'test': accs}
    model.log_results.append(log_obj)
    model.wblogger(log_obj)
