# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import inspect
import importlib
from argparse import Namespace
from torch.nn import Module


def get_all_backbones():
    return [module.split('.')[0] for module in os.listdir('backbones')
            if not module.find('__') > -1 and 'py' in module]


NAMES = {}
for backbone in get_all_backbones():
    mod = importlib.import_module(f'backbones.' + backbone)
    backbone_classes_name = [x for x in mod.__dir__() if
                             'type' in str(type(getattr(mod, x))) and 'Module' in str(
                                 inspect.getmro(getattr(mod, x))[1:])]
    for d in backbone_classes_name:
        c = getattr(mod, d)
        if hasattr(c, 'NAME'):
            NAMES[c.NAME] = c


def get_backbone(args: Namespace) -> Module:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.backbone in NAMES.keys()
    return NAMES[args.backbone](args)
