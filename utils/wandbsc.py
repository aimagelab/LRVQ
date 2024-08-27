# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import wandb
from argparse import Namespace
from utils import random_id


def innested_vars(args: Namespace):
    new_args = vars(args).copy()
    for key, value in new_args.items():
        if isinstance(value, Namespace):
            new_args[key] = innested_vars(value)
    return new_args


class WandbLogger:
    def __init__(self, args: Namespace, prj, entity, name=None, run_id=None):
        self.active = args.wandb
        if run_id is None:
            number_of_alphabets = np.random.randint(5, 10)
            self.run_id = random_id(number_of_alphabets)
            if name is not None:
                name += f'-{self.run_id}'
        else:
            self.run_id = name = run_id
        self.name = name
        if self.active:
            wandb.init(project=prj, entity=entity, config=innested_vars(args), name=name)

    def __call__(self, obj: any, **kwargs):
        if wandb.run:
            wandb.log(obj, **kwargs)
