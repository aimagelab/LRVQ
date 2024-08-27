# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import socket
import random
from time import sleep
from typing import Tuple, Optional

import torch
import numpy as np
import os
local = False if 'ssh_connection' in os.environ else True
MY_PATH = '../TraGen/data' #None # './data/'
if MY_PATH is None:
    raise ValueError('Please set MY_PATH in utils/conf.py')


def get_device(jobs_per_gpu=10) -> torch.device:
    """
    returns the gpu device if available else cpu.
    """
    if socket.gethostname() and torch.cuda.is_available():
        while True:
            lines = np.array(os.popen('nvidia-smi | grep " c " | awk \'{print $2}\'').read().splitlines()).astype(int)
            unique, counts = np.unique(lines, return_counts=True)
            if len(unique) > 1 and np.min(counts) < jobs_per_gpu:
                return torch.device('cuda:{}'.format(np.argmin(counts)))
            elif len(unique) == 0:
                return torch.device('cuda:0')
            elif len(unique) == 1:
                return torch.device('cuda:{}'.format('0' if unique.item() == 1 else '1'))
            sleep((random.random() + 1) * 5)
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def base_path() -> Optional[str]:
    """
    returns the base bath where to log accuracies and tensorboard data.
    """
    return MY_PATH


def base_path_dataset() -> Optional[str]:
    """
    returns the base bath where to log accuracies and tensorboard data.
    """
    return MY_PATH


def set_random_seed(seed: int) -> None:
    """
    sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
