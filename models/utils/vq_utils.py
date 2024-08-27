# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, List

import torch
from torch import Tensor

from utils.loss import MinimumOverNLoss
from utils.scheduler import BaseScheduler
from utils.utilities import get_module
import torch.nn.functional as F


class VariationalLoss(torch.nn.Module):
    def __init__(self, args):
        super(VariationalLoss, self).__init__()
        self.args = args
        self.PreLoss = MinimumOverNLoss()

    def forward(self, out: Tuple[List[dict], dict], tgt: Tensor):
        preds = out['pred'].reshape(self.args.generated_samples, out['pred'].shape[0] // self.args.generated_samples,
                                    out['pred'].shape[1], out['pred'].shape[2])
        pre_loss = self.PreLoss(preds, tgt)

        return {'pred_loss': pre_loss, 'embedding_loss': out['embedding_loss']}


class SchedulerGumbelTemp(BaseScheduler):
    def __init__(self, scheduler_name, stepper, scheduler_params):
        super().__init__(scheduler_name, stepper, scheduler_params)

    def on_batch_start(self, module, *args, **kwargs):
        # enter in scheduler function with starting_iter, max_iter, starting_value, ending_value, and current_step
        t = self.step(module)
        get_module(module).variational_quantization.tau = t
        module.args.gumbel_temp = t


class SchedulerAlphaEmbedding(BaseScheduler):
    def __init__(self, scheduler_name, stepper, scheduler_params):
        super().__init__(scheduler_name, stepper, scheduler_params)

    def on_batch_start(self, module, *args, **kwargs):
        t = self.step(module)
        module.alpha_embedding = t
        module.args.alpha_embedding = t

    def on_training_end(self, module, *args, **kwargs):
        # at the training end, the scheduler will be called to set the final value
        t = self.step(module)
        module.alpha_embedding = t
        module.args.alpha_embedding = t


class SchedulerAlphaLoraEmbedding(BaseScheduler):
    def __init__(self, scheduler_name, stepper, scheduler_params):
        super().__init__(scheduler_name, stepper, scheduler_params)
        assert scheduler_params[-1] in [0.25, 1], 'Maximum value of lora_alpha must be 0.25 or 1'
        self.update_opt = False

    def on_batch_start(self, module, *args, **kwargs):
        t = self.step(module)
        get_module(module).variational_quantization.lora_alpha = t
        get_module(module).variational_quantization.scaling = t
        if not self.update_opt:
            if t > 0 and module.args.freeze_static_vq:
                get_module(module).variational_quantization.embedding.requires_grad = False
                # remove from the optimizer the embedding parameters
                module.opt.param_groups[0]['params'] = [param for param in module.opt.param_groups[0]['params'] if
                                                        param is not get_module(module).variational_quantization.embedding.weight]
                module.opt.state = {k: v for k, v in module.opt.state.items() if k
                                    is not get_module(module).variational_quantization.embedding.weight}
                # do the same on the state_dict
                self.update_opt = True

    def on_training_end(self, module, *args, **kwargs):
        # at the training end, the scheduler will be called to set the final value
        t = self.step(module)
        get_module(module).variational_quantization.lora_alpha = t
        get_module(module).variational_quantization.scaling = t


class FreezeAndLoadPixel:
    def __init__(self, module):
        self.pixelcnn_state_dict = module.net.pixelcnn.state_dict()

    def on_epoch_end(self, module, *args, **kwargs):
        get_module(module).pixelcnn.load_state_dict(self.pixelcnn_state_dict)
        for param in get_module(module).pixelcnn.parameters():
            param.requires_grad = True

    def after_training_epoch(self, module, *args, **kwargs):
        self.pixelcnn_state_dict = get_module(module).pixelcnn.state_dict()


class MyCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, use_gumbel=False, *args, **kwargs):
        self.use_gumbel = use_gumbel
        super(MyCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, out, target):
        out = out.view(-1, out.shape[-1])
        target = target.view(-1).long()
        if self.use_gumbel:
            return F.nll_loss(torch.log(F.gumbel_softmax(out, 1)), target)
        return super(MyCrossEntropyLoss, self).forward(out, target)
