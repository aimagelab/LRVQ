# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
from functools import partial

import torch
from utils.conf import set_random_seed
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from backbones import NAMES as BACKBONE_NAMES
from models import get_all_models
from utils.utilities import parse_bool, parse_dict


def parse_args():
    parser = ArgumentParser(description='LRVQ', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--backbone', type=str, choices=BACKBONE_NAMES, help='Backbone name.', required=True)
    # torch.set_num_threads(4)
    args = parser.parse_known_args()[0]

    mod = importlib.import_module('models.' + args.model)
    dataset = importlib.import_module('datasets.' + args.dataset)
    backbone = importlib.import_module('backbones.' + args.backbone)

    add_management_args(parser)
    add_experiment_args(parser)
    add_optimization_args(parser)
    parser = getattr(mod, 'get_parser')(parser)
    parser = getattr(dataset, 'get_parser')(parser)
    parser = getattr(backbone, 'get_parser')(parser)

    args = parser.parse_args()

    if args.set_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.set_device)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.save_checks:
        assert args.save_every % args.validate_every == 0, 'save_every must be a multiple of validate_every'

    return args


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """

    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_steps', type=lambda s: [] if s == '' else [int(v) for v in s.split(',')],
                        default='', help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Eval Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='multistep',
                        choices=['multistep'])
    parser.add_argument('--fp16', type=parse_bool,  default=False,)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--wandb', type=parse_bool,  default=False,
                        help='Enable wandb logging')
    parser.add_argument('--wb_prj', type=str, default='LRVQ',
                        help='Wandb project')
    parser.add_argument('--job_id', type=int, default=os.environ['SLURM_JOBID']
                        if 'SLURM_JOBID' in os.environ else 0,
                        help='Job id')
    parser.add_argument('--wb_entity', type=str,
                        help='Wandb entity')
    parser.add_argument('--custom_log', type=parse_bool,  default=False,
                        help='Enable log (custom for each model, must be implemented)')
    parser.add_argument('--save_checks', type=parse_bool,  default=False,
                        help='Save checkpoints')
    parser.add_argument('--validate_every', type=int, default=10, )
    parser.add_argument('--save_every', type=int, default=10, )
    parser.add_argument('--set_device', default=None, type=str, nargs='+')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_modules_to_exclude', nargs='+', default=[])
    parser.add_argument('--only_eval', type=parse_bool,  default=False)
    parser.add_argument('--generated_samples', type=int, default=20)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--args_to_update', nargs='+', default=[])
    parser.add_argument('--reinit_wandb', type=parse_bool,  default=False, help='Reinitialize wandb')
    parser.add_argument('--wandb_postfix', type=str, default=None, help='Postfix for wandb run name')
    parser.add_argument('--logger_name', type=str, default=None, help='Name of the logger')
    parser.add_argument('--checkpoint_metric', type=str, default='ade_tf_val', help='Metric to use for checkpointing')
    parser.add_argument('--reset_checkpoint_epoch', type=parse_bool,  default=False, help='Reset checkpoint epoch')
    parser.add_argument('--resume_training', type=parse_bool,  default=False, help='Resume training')
    parser.add_argument('--n_reduced_samples', type=int, default=None, help='Number of reduced samples')
    parser.add_argument('--reduce_sampling_method',
                        type=partial(parse_dict, params=['method', 'params']),
                        default={}, help='Reduce sampling method')


def add_optimization_args(parser: ArgumentParser) -> None:
    args = parser.parse_known_args()[0]
    optimizer = getattr(torch.optim, args.optimizer)
    for arg in optimizer.__init__.__code__.co_varnames:
        if arg not in ['self', 'args', 'kwargs', 'params', 'lr']:
            parser.add_argument(f'--opt_{arg}')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
