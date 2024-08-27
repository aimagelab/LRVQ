# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import inspect
import importlib
from datasets.utils.base_dataset import BaseDataset
from argparse import Namespace


def get_all_datasets():
    return [dataset.split('.')[0] for dataset in os.listdir('datasets')
            if not dataset.find('__') > -1 and 'py' in dataset]


NAMES = {}
for dataset in get_all_datasets():
    mod = importlib.import_module('datasets.' + dataset)
    dataset_classes_name = [x for x in mod.__dir__() if
                            'type' in str(type(getattr(mod, x))) and 'BaseDataset' in str(
                                inspect.getmro(getattr(mod, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c


def get_dataset(args: Namespace) -> BaseDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
