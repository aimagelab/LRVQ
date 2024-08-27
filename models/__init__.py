# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}.get(model.replace('_', ''))
    if class_name is not None:
        names[model] = getattr(mod, class_name)


def get_model(args, backbone, loss, dataset):
    return names[args.model](backbone, loss, args, dataset)
