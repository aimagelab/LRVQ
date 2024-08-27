# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import socket
import sys

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(os.path.join(conf_path, 'utils'))

from utils.conf import base_path
from utils.wandbsc import WandbLogger
import uuid
from datasets import get_dataset
from models import get_model
from utils.args import parse_args
from utils.training import train
from utils.utilities import load_checkpoint, set_logger, print_arguments
import logging

set_logger(console=True)

logger = logging.getLogger(__name__)


def main(args=None) -> None:
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    args = load_checkpoint(args)
    printable_args = {k: v for k, v in vars(args).items() if k != 'checkpoint'}
    print_arguments(printable_args)

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset)

    if args.job_id == 0:
        run_id = model.args.name if model.args.name is not None and not model.args.reinit_wandb else None
    else:
        run_id = str(args.job_id)
    if model.args.name is None:
        model.args.name = model.NAME
    if model.args.wandb_postfix is not None:
        model.args.name += '_' + model.args.wandb_postfix
    model.wblogger = WandbLogger(model.args, name=model.args.name, prj=model.args.wb_prj,
                                 entity=model.args.wb_entity,
                                 run_id=run_id)
    model.args.name = model.wblogger.name
    model.result_dir = os.path.join(base_path(), 'results', model.dataset_name, model.NAME, model.wblogger.name)
    if model.args.save_checks:
        os.makedirs(model.result_dir, exist_ok=True)
        print(f'Writing results in {model.result_dir}')
    print('Total number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    train(model, dataset, args)


if __name__ == "__main__":
    main()
