# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import os
from collections import defaultdict
import pickle
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from utils.status import progress_bar
from utils.utilities import final_displacement_error, average_displacement_error, get_denormalize_fun
from utils.subsample import Subsample


@torch.no_grad()
def evaluate(model, loader, epoch, teacher_forcing=False):
    # try to get denormalize function from dataset
    denormalize = get_denormalize_fun(loader)
    model.eval()
    total_loss, ade, fde, total_traj, ade_per_token = 0., 0., 0., 0., torch.zeros(model.args.pred_len).to(model.device)
    teacher_forcing = True if teacher_forcing else False
    n_samples = model.args.generated_samples if not teacher_forcing else 1
    if not teacher_forcing and model.args.n_reduced_samples is not None:
        n_reduced_samples = model.args.n_reduced_samples
    else:
        n_reduced_samples = n_samples
    sub_sampler = Subsample(**model.args.reduce_sampling_method, n_samples=n_reduced_samples)
    ades, fdes = defaultdict(float), defaultdict(float)
    ades_lst, fdes_lst = [], []
    accs = 0.
    for batch_idx, data in enumerate(loader):
        data = [tensor.to(model.device) for tensor in data]
        try:
            src_abs, tgt_abs, seq_start_end, src, tgt, index = data
        except ValueError:
            raise ValueError(f'Wrong number of arguments in data: {data}, '
                             f'collate function of validation and test dataloader should outputs'
                             f' src_abs, tgt_abs, seq_start_end, src, tgt, index')
        _, src_len, _ = src.shape
        generated = []
        ade_batch, fde_batch, loss_batch, loss_batch_tf, index_batch, z_batch, distances, acc = [], [], [], [], [], \
            [], [], []
        for i in range(n_samples):
            p = 'autoregressive' if not teacher_forcing else 'teacher_forcing'
            outs = model(src, tgt, seq_start_end, [p])
            out = outs[p]
            pred_abs = denormalize(out['pred_abs'], seq_start_end, index)
            generated.append(pred_abs)
            ade_batch.append(average_displacement_error(pred_abs, tgt_abs, mode='raw').sum(dim=1))
            fde_batch.append(final_displacement_error(pred_abs[:, -1], tgt_abs[:, -1], mode='raw'))
            loss_batch.append(out['loss'])
            index_batch.append(out['indices'])
            z_batch.append(out['z'])
            distances.append(average_displacement_error(pred_abs, tgt_abs, mode='raw'))
            if p == 'teacher_forcing':
                acc.append(out['acc'])
        total_traj += src_abs.shape[0]
        ade_batch = torch.stack(ade_batch)
        fde_batch = torch.stack(fde_batch)
        loss_batch = torch.stack(loss_batch)
        generated = torch.stack(generated)
        acc = torch.stack(acc) if len(acc) > 0 else None
        # Loss
        subsampled_idx = sub_sampler(generated)
        subsampled_idx_pr = subsampled_idx.permute(1, 2, 3, 0)
        # recalculate metrics for the subsampled predictions
        nn = subsampled_idx_pr.shape[-1]
        ade_batch_subsampled = average_displacement_error(subsampled_idx_pr,
                                                          tgt_abs[..., None].repeat(1, 1, 1, nn),
                                                          mode='raw').permute(2, 0, 1).sum(dim=2)
        fde_batch_subsampled = final_displacement_error(subsampled_idx_pr[:, -1],
                                                        tgt_abs[..., None].repeat(1, 1, 1, nn)[:, -1],
                                                        mode='raw').permute(1, 0)
        distances = average_displacement_error(subsampled_idx_pr,
                                               tgt_abs[..., None].repeat(1, 1, 1, nn),
                                               mode='raw').permute(2, 0, 1)
        total_loss += torch.min(loss_batch, dim=0)[0].sum()
        ade += torch.min(ade_batch_subsampled, dim=0)[0].sum()
        fde += torch.min(fde_batch_subsampled, dim=0)[0].sum()
        ades_lst += (torch.min(ade_batch_subsampled, dim=0)[0] / model.args.pred_len).tolist()
        fdes_lst += (torch.min(fde_batch_subsampled, dim=0)[0]).tolist()
        accs += acc.min() if acc is not None else 0
        if ade_batch_subsampled.shape[0] == 20:
            # if we have 20 samples, we can calculate metrics only for the sub-k (k=1, 5, 10, 15, 20) samples
            for myk in [1, 5, 10, 15, 20]:
                ades[f'{myk}_k'] += torch.min(ade_batch_subsampled[:myk], dim=0)[0].sum() / model.args.pred_len
                fdes[f'{myk}_k'] += torch.min(fde_batch_subsampled[:myk], dim=0)[0].sum()
            if 'nba' in model.args.dataset:
                for time_i in range(1, 5):
                    ades[f'{time_i}_{20}_k'] += (distances[..., :5 * time_i]).mean(dim=-1).min(dim=0)[0].sum()
                    fdes[f'{time_i}_{20}_k'] += (distances[..., 5 * time_i - 1]).min(dim=0)[0].sum()
            if 'nfl' in model.args.dataset:
                # create ades and fdes for each sample every 5 time steps
                for time_i in [5, 10, 16]:
                    ades[f'{time_i / 5}_{20}_k'] += (distances[..., :time_i]).mean(dim=-1).min(dim=0)[0].sum()
                    fdes[f'{time_i / 5}_{20}_k'] += (distances[..., time_i - 1]).min(dim=0)[0].sum()
        progress_bar(batch_idx, len(loader), epoch, (total_loss / total_traj).item(),
                     ade=(ade / total_traj / model.args.pred_len).item(),
                     fde=(fde / total_traj).item(),
                     batch=f'{batch_idx + 1}/{len(loader)}',
                     ade_median=np.median(ades_lst),
                     fde_median=np.median(fdes_lst),
                     )

    ade_val = ade / (total_traj * model.args.pred_len)
    # FDE
    fde_val = fde / total_traj
    for name in ades.keys():
        ades[name] /= total_traj
        fdes[name] /= total_traj

    accs /= total_traj * model.args.pred_len
    scaling_factor = getattr(loader.dataset, 'traj_scale', 1)
    ade_val *= scaling_factor
    fde_val *= scaling_factor
    for name in ades.keys():
        ades[name] *= scaling_factor
        fdes[name] *= scaling_factor

    total_loss = total_loss / total_traj
    ade_median = np.median(ades_lst) * scaling_factor
    fde_median = np.median(fdes_lst) * scaling_factor
    model.train()
    print('Total trajectories: ', total_traj)
    postfix = '' if not teacher_forcing else '_tf'
    output_metrics = {f'ade{postfix}': ade_val, f'fde{postfix}': fde_val, f'loss{postfix}': total_loss,
                      f'ade_median{postfix}': ade_median, f'fde_median{postfix}': fde_median, f'acc{postfix}': accs}
    try:
        fig = plot_generated(src_abs, tgt_abs, generated, torch.min(ade_batch, dim=0)[1],
                             show=True if not model.args.wandb else False)
        output_metrics[f'figure_samples{postfix}'] = wandb.Image(fig)
    except:
        pass
    for name, value in ades.items():
        output_metrics[f'ade_{name}{postfix}'] = value
    for name, value in fdes.items():
        output_metrics[f'fde_{name}{postfix}'] = value
    if model.args.only_eval:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # dump the metrics
        with open(os.path.join(model.result_dir, f'metrics_{now}.pkl'), 'wb') as f:
            pickle.dump(output_metrics, f)
    return output_metrics


def plot_generated(src, tgt, pred, index=None, show=True, which=None):
    number_of_plots = min(3, src.shape[0])
    fig, ax = plt.subplots(number_of_plots, 1, figsize=(10, 5))
    for i, a in enumerate(ax):
        if which is not None:
            rand = which[i]
        else:
            rand = np.random.choice(pred.shape[1])
        line = a.plot(src[rand, :, 0].cpu(), src[rand, :, 1].cpu(), color='darkgreen', marker='o')[0]
        add_arrow(line)
        line = a.plot(tgt[rand, :, 0].cpu(), tgt[rand, :, 1].cpu(), color='darkblue', marker='o')[0]
        add_arrow(line)
        legend = ['src', 'tgt']

        if index is not None:
            idx = index[rand]
            a.plot(pred[idx, rand, :, 0].detach().cpu(), pred[idx, rand, :, 1].detach().cpu(),
                   color='darkred',
                   marker='o')

            legend += ['best']
        for p in range(pred.shape[0]):
            a.plot(pred[p, rand, :, 0].detach().cpu(), pred[p, rand, :, 1].detach().cpu(), color='pink', alpha=0.6, marker='o',
                   zorder=0)
        legend += ['generated']
        a.legend(legend)
    if show:
        plt.show()
    else:
        return fig


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )
