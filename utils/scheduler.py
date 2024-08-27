# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from abc import abstractmethod
from functools import partial


def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0)))  # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2)  # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0  # interpolate accordingly
    return t


def frange_linear_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a linear schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0)))  # what fraction of the way through are we
    t = alpha * t1 + (1 - alpha) * t0  # interpolate accordingly
    return t


NAMES = {'frange_linear_anneal': frange_linear_anneal,
         'cos_anneal': cos_anneal}


class BaseScheduler:
    def __init__(self, scheduler_name, stepper, scheduler_params):
        self.stepper = stepper
        self.scheduler_params = scheduler_params
        self.scheduler = partial(NAMES[scheduler_name], *scheduler_params)

    def step(self, module):
        # enter in scheduler function with starting_iter, max_iter, starting_value, ending_value, and current_step
        step = getattr(module, self.stepper)
        t = self.scheduler(step)
        return t

    @abstractmethod
    def on_batch_start(self, module, *args, **kwargs):
        # overwrite module's value with the scheduler's value
        # ATTENTION!! remember to overwrite the module.args parameter otherwise it will not be save in the checkpoint
        pass

    @abstractmethod
    def on_training_end(self, module, *args, **kwargs):
        # overwrite module's value with the scheduler's value
        # ATTENTION!! remember to overwrite the module.args parameter otherwise it will not be save in the checkpoint
        pass
