# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
from torch import Tensor


class PredictionLoss(nn.Module):
    def __init__(self):
        super(PredictionLoss, self).__init__()
        self.L2 = nn.PairwiseDistance(p=2)

    def forward(self, prediction: Tensor, target: Tensor, reduce=None, mask: Tensor = None) -> Tensor:
        if mask is not None:
            target = target[mask.bool()]
            prediction = prediction[mask.bool()]
            B, C = target.shape
        else:
            B, T, C = target.shape
        # apply mask
        pred_loss = self.L2(prediction[..., :C].contiguous().view(-1, C), target.contiguous().view(-1, C))
        if mask is None:
            pred_loss = pred_loss.reshape(B, T)
        binary_col_loss = torch.zeros(pred_loss.shape).to(pred_loss.device)

        if reduce is None:
            pred_loss = pred_loss.mean()
            binary_col_loss = prediction[..., C].abs().mean() if C != prediction.shape[-1] else binary_col_loss.mean()
        elif reduce == 'raw':
            pred_loss = pred_loss.mean(1)
            binary_col_loss = prediction[..., C].abs().mean(1) if C != prediction.shape[-1] else binary_col_loss.mean(1)
        else:
            raise ValueError("Invalid reduce value")

        return pred_loss + binary_col_loss

    def __call__(self, *args, **kwargs):
        return super(PredictionLoss, self).__call__(*args, **kwargs)


class MinimumOverNLoss(nn.Module):
    def __init__(self, agg='min'):
        super(MinimumOverNLoss, self).__init__()
        self.L2 = nn.PairwiseDistance(p=2)
        self.idxes = None
        self.agg = agg

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        B, T, C = target.shape
        N = prediction.shape[0]
        # repeat target
        pred_loss = self.L2(prediction[..., :C].contiguous().view(-1, C),
                            target.unsqueeze(0).repeat(N, 1, 1, 1).contiguous().view(-1, C)).reshape(N, B, T).mean(-1)
        if self.agg == 'min':
            pred_loss, idxes = pred_loss.min(0)
        elif self.agg == 'mean':
            pred_loss = pred_loss.mean(0)
            idxes = None
        elif self.agg == 'max':
            pred_loss, idxes = pred_loss.max(0)
        else:
            raise ValueError(f'Invalid aggregation type {self.agg}')
        # get prediction loss indexing with the minimum loss
        self.idxes = idxes
        binary_col_loss = torch.zeros(pred_loss.shape).to(pred_loss.device)

        pred_loss = pred_loss.mean()
        binary_col_loss = prediction[..., C].abs().mean() if C != prediction.shape[-1] else binary_col_loss.mean()

        return pred_loss + binary_col_loss

    def __call__(self, *args, **kwargs):
        return super(MinimumOverNLoss, self).__call__(*args, **kwargs)


class FourierDistanceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
            self, inputs: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = 1
    ) -> torch.Tensor:
        norm_factor = math.sqrt(inputs.shape[-2] / 2)
        fft_inputs = torch.fft.rfft(inputs, dim=-2) / norm_factor
        fft_target = torch.fft.rfft(target, dim=-2) / norm_factor
        return (
                weight
                * torch.sqrt(
            (fft_target.real - fft_inputs.real).pow(2)
            + (fft_target.imag - fft_inputs.imag).pow(2)
            + self.epsilon
        )
                .reshape(fft_inputs.shape[0], -1)
                .mean(-1)
        ).mean()
