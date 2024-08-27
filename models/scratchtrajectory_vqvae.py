# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Union
from datasets import BaseDataset
from models.scratchtrajectory import ScratchTrajectory
import torch

from models.utils.vq_utils import MyCrossEntropyLoss, FreezeAndLoadPixel, SchedulerAlphaEmbedding, \
    SchedulerAlphaLoraEmbedding
from utils.eval import evaluate

from utils.status import progress_bar
from utils.utilities import attention_mask, unrotate_and_unshift_batch, denormalize_data, \
    parse_bool, get_module
from backbones import NAMES as BACKBONE_NAMES
from torch.functional import F


def get_parser(parser) -> ArgumentParser:
    """
    Defines the arguments of the model.
    """
    # data args
    parser.add_argument('--avoid_last_obs', type=parse_bool, default=False,
                        help='Avoid last obs (0,0) after normalization')
    parser.add_argument('--random_rotation', default=True, type=parse_bool, help='Random rotation')
    parser.add_argument('--random_rotation_degrees', default=None, type=float, help='Random rotation degrees')
    parser.add_argument('--rotation_type', type=str, default='normal', help='Rotation type', choices=['normal',
                                                                                                      'red'])
    parser.add_argument('--spatial_att_adj', type=parse_bool, default=False, help='Use spatial attention mask with adj')
    parser.add_argument('--space_threshold', default=5, type=float, help='Space threshold for spatial attention mask')
    parser.add_argument('--validate_with_tf', type=parse_bool, default=False, help='Validate with teacher forcing')
    parser.add_argument('--apply_normalization', type=parse_bool, default=False, help='Apply normalization')
    parser.add_argument('--nabs', type=parse_bool, default=True, help='Use nabs')

    # vq training args
    parser.add_argument('--alpha_embedding', default=1, type=float, help='Alpha embedding loss')
    parser.add_argument('--alpha_embedding_annealing', type=lambda x: x,
                        help='Parameters for alpha embedding annealing',
                        default=[], nargs='+')
    parser.add_argument('--lora_alpha_annealing', type=lambda x: x,
                        help='Parameters for lora_alpha annealing',
                        default=[], nargs='+')
    parser.add_argument('--fft_before_loss', type=parse_bool, default=True)  #

    # pixelcnn
    # initialization args
    parser.add_argument('--reinit_pixelcnn', type=parse_bool, default=False, help='Reinitialize PixelCNN')
    parser.add_argument('--custom_pixelcnn', type=parse_bool, default=False,
                        help='Avoid reload PixelCNN if checkpoint is provided')
    parser.add_argument('--finetune_pixelcnn', type=parse_bool, default=False)
    parser.add_argument('--use_saved_pixelcnn', type=parse_bool, default=False, help='Use saved PixelCNN')
    parser.add_argument('--pixelcnn_from_different_path', type=str, default=None, help='Path to load pixelcnn from')

    # training and evaluation args
    parser.add_argument('--pixelcnn_lr', default=0.0005, type=float, help='PixelCNN learning rate')  #
    parser.add_argument('--pixelcnn_n_epochs', default=1000, type=int, help='PixelCNN n epochs')
    parser.add_argument('--pixelcnn_train_with_original_src', type=parse_bool, default=False, )
    parser.add_argument('--pixelcnn_eval_ade_during_training', type=parse_bool, default=True, )
    parser.add_argument('--pixelcnn_scheduler_name', type=str, default='multistep',
                        help='PixelCNN scheduler', choices=['multistep',
                                                            'multistep2'])
    parser.add_argument('--pixelcnn_opt', type=str, default='adamw', help='PixelCNN optimizer', )
    parser.add_argument('--rotate_trj_pixelcnn', type=parse_bool, default=True)
    parser.add_argument('--freeze_static_vq', type=parse_bool, default=False)
    return parser


class ScratchTrajectoryVQVAE(ScratchTrajectory):
    NAME = 'scratchtrajectory_vqvae'
    POSSIBLE_BACKBONES = ['vqvae']
    POSSIBLE_BACKBONES_MODULE = [BACKBONE_NAMES[backbone] for backbone in POSSIBLE_BACKBONES]
    ANNEALING_CALLBACKS = {'alpha_embedding': SchedulerAlphaEmbedding,
                           'lora_alpha': SchedulerAlphaLoraEmbedding}

    def __init__(self, backbone: POSSIBLE_BACKBONES_MODULE, loss, args, transform):
        self.lsp_step = 0
        self.lsp_metrics = {}
        self.pixelcnn_scheduler = None
        self.pixelcnn_opt = None
        self.test_samples = None
        self.train_samples = None
        assert backbone.NAME in self.POSSIBLE_BACKBONES
        super(ScratchTrajectoryVQVAE, self).__init__(backbone, loss, args, transform)
        self.pixelcnn_step = 0
        self.alpha_embedding = self.args.alpha_embedding
        for scheduler in self.ANNEALING_CALLBACKS.keys():
            annealing = getattr(self.args, f'{scheduler}_annealing')
            if len(annealing):
                if annealing[3] == 'max':
                    annealing[3] = self.total_steps
                self.callbacks.append(self.ANNEALING_CALLBACKS[scheduler](annealing[0],
                                                                          annealing[1],
                                                                          [float(x) for x in annealing[2:]]))
        self.callbacks.append(FreezeAndLoadPixel(self))

        self.pixelcnn_metrics = {}

        if getattr(self.args, 'optimizer_state_dict', None) is not None:
            for param_gr in self.args.optimizer_state_dict['param_groups']:
                for key in param_gr.keys():
                    if isinstance(param_gr[key], torch.Tensor):
                        param_gr[key] = param_gr[key].to(self.device)
            try:
                self.opt.load_state_dict(self.args.optimizer_state_dict)
            except:
                print('WARNING: Cannot load optimizer state dict')
            # send all the states of the optimizer to the device
            for key in self.opt.state_dict()['state'].keys():
                for k in self.opt.state_dict()['state'][key].keys():
                    if isinstance(self.opt.state_dict()['state'][key][k], torch.Tensor):
                        self.opt.state_dict()['state'][key][k] = self.opt.state_dict()['state'][key][k].to(self.device)

        # self.scheduler = self.set_scheduler(self.opt)

    def load_checkpoint_states_dict(self, ckp, *args, **kwargs):
        state_dict = ckp['model_state_dict']
        if self.args.custom_pixelcnn and not self.args.use_saved_pixelcnn:
            for key in list(state_dict.keys()):
                if 'pixelcnn' in key:
                    state_dict.pop(key)
        model_state = get_module(self).state_dict()
        model_state.update(state_dict)
        try:
            get_module(self).load_state_dict(model_state)
        except RuntimeError:
            print('WARNING: Cannot load model state dict, trying to load with strict=False')
            get_module(self).load_state_dict(model_state, strict=False)
        self.ade_val = ckp.get('best_ade', float('inf'))

    def observe(self, src, tgt, seq_start_end):
        wb_log = {}
        self.train()
        self.opt.zero_grad()

        src = src.requires_grad_(True)
        tgt = tgt.requires_grad_(True)

        (encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_norm, shift_value,
         theta, means) = self.preprocess(src, tgt, seq_start_end, spatial_att_adj=self.args.spatial_att_adj,
                                         rotate=self.args.random_rotation, degrees=self.args.random_rotation_degrees)
        inputs = {'src': encoder_input,
                  'tgt': decoder_input,
                  'src_mask': src_mask,
                  'tgt_mask': tgt_mask,
                  'space_mask': space_mask,
                  'tgt_space_mask': tgt_space_mask,
                  'tgt_enc': tgt_norm}

        outs = get_module(self)(**inputs)
        preds = outs['pred']

        pred_loss = self.loss(preds, tgt_norm, mask=outs.get('masked_token'))

        wb_log.update({'pred_loss': pred_loss.item()})
        #
        # if self.args.fft_before_loss:
        #     pred_loss = FourierDistanceLoss()(preds, tgt_norm)
        #     wb_log.update({'pred_loss_fft': pred_loss.item()})

        loss = pred_loss + self.alpha_embedding * outs[
            'embedding_loss']

        wb_log.update({'embedding_loss': outs['embedding_loss'].item(),
                       'perplexity': outs['perplexity'].item(),
                       'alpha_embedding': self.alpha_embedding,
                       'median_delta_embedding': (outs['indices'].diff(1) != 0).float().mean(1).median().item()
                       if outs['indices'].dim() == 2 else 0,
                       'lora_alpha': get_module(self).variational_quantization.lora_alpha if hasattr(
                           get_module(self).variational_quantization, 'lora_alpha') else 0,
                       })

        loss.backward()
        self.opt.step()
        self.wb_log.update(wb_log)
        self.opt.zero_grad()

        return loss.item()

    def on_test_start(self, dataset, *args, **kwargs):
        super(ScratchTrajectoryVQVAE, self).on_test_start(dataset, *args, **kwargs)
        self.args.pixelcnn_epoch_start = 0
        if self.args.pixelcnn_from_different_path is None:
            if not self.args.use_saved_pixelcnn:
                if self.args.finetune_pixelcnn:
                    self.load_pixelcnn(os.path.join(os.path.dirname(self.args.checkpoint_path), 'last_pixelcnn.pt'))
                total_epochs = self.args.pixelcnn_n_epochs
                train_loader, valid_loader, test_loader, train_sampler = dataset.get_data_loaders()
                for i in range(self.args.pixelcnn_epoch_start // self.args.validate_every,
                               total_epochs // self.args.validate_every):
                    self.args.pixelcnn_epoch_start = i * self.args.validate_every
                    self.args.pixelcnn_epoch_end = (i + 1) * self.args.validate_every
                    self.pixelcnn_train_and_test(train_loader, valid_loader, test_loader, train_sampler)
                    if self.args.pixelcnn_eval_ade_during_training:
                        metrics = evaluate(self, test_loader, self.args.pixelcnn_epoch_end - 1)
                        self.wblogger({'pixelcnn_train.test': metrics,
                                       'pixelcnn_epoch': self.args.pixelcnn_epoch_end - 1})
                        if metrics.get('ade') <= self.pixelcnn_metrics.get('ade', float('inf')):
                            self.pixelcnn_metrics.update(metrics)
                            self.save_pixelcnn(self.args.pixelcnn_epoch_end - 1)
                            self.save_checkpoint(name='best_pixel', metric=metrics.get('ade'))
                        self.save_pixelcnn(self.args.pixelcnn_epoch_end - 1, 'last')
                        self.save_checkpoint(name='last_pixel', metric=metrics.get('ade'))
                    self.args.reinit_pixelcnn = False
                # save the last pixelcnn model
                self.save_pixelcnn(total_epochs, 'final')
                if self.args.save_checks:
                    self.load_pixelcnn()
                # reinitialize the pixelcnn parameters
                self.pixelcnn_scheduler = None
                self.pixelcnn_opt = None
        else:
            self.load_pixelcnn(self.args.pixelcnn_from_different_path)

    def pixelcnn_train_and_test(self, train_loader, valid_loader, test_loader, train_sampler):
        # train PIXELCNN1D on all the generated samples
        get_module(self).eval()

        # freeze all the parameters of the model except for the pixelcnn
        for param in get_module(self).parameters():
            param.requires_grad = False
        #
        # reinitialize the pixelcnn parameters
        if self.args.reinit_pixelcnn:
            get_module(self).init_pixelcnn(self.args.pixelcnn,
                                           init_weights=False)
            get_module(self).to(self.device)
            params = get_module(self).pixelcnn.named_parameters()
            get_module(self).pixelcnn.init_weights([p for n, p in params if 'encoder_layers' not in n])

        for name, param in get_module(self).pixelcnn.named_parameters():
            param.requires_grad = True

        print(
            f'Pixelcnn parameters number: '
            f'{sum(p.numel() for p in get_module(self).pixelcnn.parameters()if p.requires_grad)}'
        )
        self.train_pixelcnn(train_loader, train_sampler, valid_loader)
        test_metrics = self.test_pixelcnn(test_loader, 'test')
        self.wblogger({
            'test':
                test_metrics,
            'epoch': self.epoch
        })

        # unfreeze all the parameters of the model
        for param in get_module(self).parameters():
            param.requires_grad = True

    def after_training_epoch(self, dataset: BaseDataset, *args, **kwargs):
        super(ScratchTrajectoryVQVAE, self).after_training_epoch(dataset, *args, **kwargs)
        # reinitialize the pixelcnn parameters
        self.pixelcnn_scheduler = None
        self.pixelcnn_opt = None

    def init_optimizer_pixelcnn(self):
        if self.pixelcnn_opt is None:
            if self.args.pixelcnn_opt == 'adam':
                self.pixelcnn_opt = torch.optim.Adam(get_module(self).pixelcnn.parameters(), lr=self.args.pixelcnn_lr)
            elif self.args.pixelcnn_opt == 'adamw':
                self.pixelcnn_opt = torch.optim.AdamW(get_module(self).pixelcnn.parameters(),
                                                      lr=self.args.pixelcnn_lr,
                                                      weight_decay=1e-2)
            else:
                raise NotImplementedError

    def init_scheduler_pixelcnn(self):
        if self.pixelcnn_scheduler is None:
            if self.args.pixelcnn_scheduler_name == 'multistep':
                self.pixelcnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.pixelcnn_opt,
                    milestones=[int(self.args.pixelcnn_n_epochs * 0.7),
                                int(self.args.pixelcnn_n_epochs * 0.9)],
                    gamma=0.8)
            elif self.args.pixelcnn_scheduler_name == 'multistep2':
                self.pixelcnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.pixelcnn_opt,
                    milestones=[int(self.args.pixelcnn_n_epochs * 2/3),
                                int(self.args.pixelcnn_n_epochs * 1/3)],
                    gamma=0.1)
            else:
                raise NotImplementedError

    def train_pixelcnn(self, train_loader, train_sampler, valid_loader):
        self.init_optimizer_pixelcnn()
        self.init_scheduler_pixelcnn()

        pixelcnn_loss = MyCrossEntropyLoss()
        pixelcnn_loss.to(self.device)
        for epoch in range(self.args.pixelcnn_epoch_start, self.args.pixelcnn_epoch_end):
            get_module(self).pixelcnn.train()
            total_loss = 0
            perplexity = 0.
            for i, data in enumerate(train_loader):
                self.pixelcnn_opt.zero_grad()
                data = [tensor.to(self.device) for tensor in data]
                src, tgt, seq_start_end = data
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                # create input for vqvae model
                inputs, outs, src_mask, space_mask, tgt_mask, tgt_space_mask, tgt_norm = self.extract_example_with_net(
                    src, tgt,
                    seq_start_end,
                    rotate=self.args.rotate_trj_pixelcnn,
                )
                perplexity += outs['perplexity'].item()
                x_pixel = outs['encoder_input'].clone() if not self.args.pixelcnn_train_with_original_src \
                    else inputs['src'].clone()
                y_pixel = outs.get('indices').clone()
                if y_pixel.dim() == 2:
                    y_pixel = y_pixel.float().unsqueeze(-1)
                y_pixel = y_pixel.permute(0, 2, 1)
                inputs = {
                    'x': x_pixel.detach(),
                    'y': y_pixel,
                    'src_mask': src_mask,
                    'space_mask': space_mask,
                    'tgt_space_mask': tgt_space_mask,
                    'return_raw_loss': True,
                    'n_repeated_elements': 1,
                }
                out = get_module(self).pixelcnn(**inputs)

                loss = out['loss']
                out = out['logits']
                loss = loss.mean()
                output_perplexity = calc_perplexity(out.argmax(-1).view(-1, 1), out.shape[-1])

                loss.backward()
                self.pixelcnn_opt.step()
                progress_bar(i, len(train_loader), epoch, loss.item(),
                             batch=f'{i + 1}/{len(train_loader)}',
                             perplexity=perplexity / (i + 1) / self.args.k_way,
                             output_perplexity=output_perplexity / self.args.k_way,
                             )
                self.wblogger({'pixelcnn_loss': loss.item(), 'pixelcnn_epoch': epoch,
                               'pixelcnn_batch': i,
                               'pixelcnn_step': self.pixelcnn_step,
                               'pixelcnn_lr': self.pixelcnn_opt.param_groups[0]['lr']})
                self.pixelcnn_step += 1
                total_loss += loss.item()
            if epoch % self.args.validate_every == 0:
                metrics = {}
                if valid_loader is not None:
                    metrics = self.test_pixelcnn(valid_loader, 'validation', epoch)
                    if metrics.get('pixelcnn_acc') >= self.pixelcnn_metrics.get('pixelcnn_acc', -float('inf')):
                        self.pixelcnn_metrics = metrics
                        self.save_pixelcnn(epoch)
                        self.save_checkpoint(name='best_pixel', metric=metrics.get('pixelcnn_acc'))
                    self.save_pixelcnn(epoch, 'last')
                    self.save_checkpoint(name='last_pixel', metric=metrics.get('pixelcnn_acc'))
                training_metrics = self.test_pixelcnn(train_loader, 'train', epoch)
                self.wblogger({
                    'training':
                        training_metrics,
                    'validation':
                        metrics,
                    'pixelcnn_epoch': epoch
                })

    def save_pixelcnn(self, epoch, name='best'):
        if self.args.save_checks:
            torch.save({
                'model': get_module(self).pixelcnn.state_dict(),
                'optimizer': self.pixelcnn_opt.state_dict(),
                'scheduler': self.pixelcnn_scheduler.state_dict(),
                'epoch': epoch,
                'step': self.pixelcnn_step,
                'metrics': self.pixelcnn_metrics
            }, os.path.join(self.result_dir, f'{name}_pixelcnn.pt'))

    def load_pixelcnn(self, path=None):
        my_path = os.path.join(self.result_dir, f'best_pixelcnn.pt') if path is None else path
        if os.path.exists(my_path):
            checkpoint = torch.load(my_path)
            get_module(self).pixelcnn.load_state_dict(checkpoint['model'])
            if self.pixelcnn_opt is None:
                self.init_optimizer_pixelcnn()
            if self.pixelcnn_scheduler is None:
                self.init_scheduler_pixelcnn()
            self.pixelcnn_opt.load_state_dict(checkpoint['optimizer'])
            self.pixelcnn_scheduler.load_state_dict(checkpoint['scheduler'])
            self.pixelcnn_step = checkpoint['step']
            self.args.pixelcnn_epoch_start = checkpoint['epoch']
            self.pixelcnn_metrics = checkpoint['metrics']

    def extract_example_with_net(self, src, tgt, seq_start_end, rotate=False):
        (encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_norm,
         shift_value, theta, means) = self.preprocess(
            src, tgt, seq_start_end, spatial_att_adj=self.args.spatial_att_adj, rotate=rotate,
            degrees=self.args.random_rotation_degrees
        )
        inputs = {'src': encoder_input, 'tgt': decoder_input, 'src_mask': src_mask,
                  'tgt_mask': tgt_mask, 'space_mask': space_mask, 'tgt_space_mask': tgt_space_mask,
                  'tgt_enc': tgt_norm}
        # make inference with vqvae model
        with torch.no_grad():
            outs = get_module(self)(**inputs, extract_training_samples=True)

        return inputs, outs, src_mask, space_mask, tgt_mask, tgt_space_mask, tgt_norm

    @torch.no_grad()
    def test_pixelcnn(self, loader, dataset_type='test', epoch=None):
        pixelcnn_loss = MyCrossEntropyLoss()
        pixelcnn_loss.to(self.device)
        get_module(self).pixelcnn.eval()
        total_preds, total_trues = [], []
        loss, acc, traj_total, perplexity = 0., 0., 0, 0.
        acc_dict = defaultdict(int)
        for i, data in enumerate(loader):
            data = [tensor.to(self.device) for tensor in data]
            if dataset_type == 'train':
                src, tgt, seq_start_end = data
            else:
                src_abs, tgt_abs, seq_start_end, src, tgt, index = data
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            # create input for vqvae model
            (inputs, outs, src_mask, space_mask, tgt_mask, tgt_space_mask, tgt_norm) = self.extract_example_with_net(
                src, tgt, seq_start_end, rotate=False)
            perplexity += outs['perplexity'].item()
            x_pixel = outs['encoder_input'].clone() if not self.args.pixelcnn_train_with_original_src \
                else inputs['src'].clone()
            y_pixel = outs.get('indices').clone().float().unsqueeze(-1)
            y_pixel = y_pixel.permute(0, 2, 1)
            input_token = torch.ones_like(y_pixel[..., :1]).to(self.device) * self.args.k_way
            input_token = input_token.detach()
            tgts = outs.get('indices')
            inputs = {
                'x': x_pixel.detach(),
                'y': input_token,
                'src_mask': src_mask,
                'space_mask': space_mask,
                'tgt_space_mask': tgt_space_mask,
            }
            # samples_test.append(inputs)
            out = get_module(self).pixelcnn(**inputs)
            loss += F.cross_entropy(out.view(-1, out.shape[-1]), tgts.view(-1), reduction='sum').item()
            acc += (out.argmax(-1) == tgts).float().sum().item()
            accs_split = (out.argmax(-1) == tgts).float().sum(0)
            inputs['y'] = torch.cat([input_token, y_pixel], dim=-1).detach()[..., :-1]
            for j in range(tgt.shape[1]):
                acc_dict[f'index_{j}'] += accs_split[j].item()
            traj_total += src.shape[0]
            total_preds.append(out.argmax(-1).cpu().numpy())
            total_trues.append(tgts.cpu().numpy())
        pixelcnn_metrics = {
            'pixelcnn_loss': loss / (traj_total * tgt.shape[1]),
            'pixelcnn_acc': acc / (traj_total * tgt.shape[1]),
        }
        print(f'pixelcnn acc dict for epoch {epoch}: ', {k: v / traj_total for k, v in acc_dict.items()})

        print(f'\nEpoch {self.epoch if epoch is None else epoch} | {dataset_type}: ' + ' '.join(
            [f'{k}: {v:.4f}' for k, v in
             pixelcnn_metrics.items()]))
        pixelcnn_metrics.update({'pixelcnn_acc_per_index': {k: v / traj_total for k, v in acc_dict.items()}})
        print(f'Perplexity on {dataset_type}: {perplexity / len(loader) / self.args.k_way}')
        return pixelcnn_metrics

    def predict_with_tf(self, encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_norm):
        out = get_module(self)(encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask,
                               tgt_enc=tgt_norm,
                               extract_training_samples=True)
        return out

    def predict_autoregressive(self, encoder_input, src_mask, space_mask, tgt_space_mask,
                               shift_value, theta, seq_start_end, batch, inpainting_tensor=None):
        start_token = torch.Tensor([0., 0., 1.]).unsqueeze(0).unsqueeze(1).repeat(batch, 1, 1).to(self.device)
        decoder_input = start_token

        tgt_mask = attention_mask(batch, self.args.pred_len, subsequent=True).to(self.device)
        out = get_module(self)(encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask,
                               inpainting_tensor=inpainting_tensor)
        return out

    def forward(self, x: torch.Tensor, tgt: Union[torch.Tensor, None], seq_start_end: torch.Tensor,
                preds_types: list = ['autoregressive']) -> dict:
        (encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_norm, shift_value, theta,
         means) = self.preprocess(x, tgt, seq_start_end, rotate=False, spatial_att_adj=self.args.spatial_att_adj)
        batch = x.shape[0]
        prediction_dict = {}
        # first validation with teacher forcing
        if 'teacher_forcing' in preds_types:
            predictions_tf = self.predict_with_tf(encoder_input, decoder_input, src_mask, tgt_mask, space_mask,
                                                  tgt_space_mask, tgt_norm)
            prediction_dict['teacher_forcing'] = predictions_tf
            x_pixel = predictions_tf['encoder_input'].clone() if not self.args.pixelcnn_train_with_original_src \
                else predictions_tf['src'].clone()
            y_pixel = predictions_tf.get('indices').clone().float().unsqueeze(-1)
            y_pixel = y_pixel.permute(0, 2, 1)
            tgts = predictions_tf.get('indices')
            inputs = {
                'x': x_pixel.detach(),
                'y': y_pixel,
                'src_mask': src_mask,
                'space_mask': space_mask,
                'tgt_space_mask': tgt_space_mask,
            }
            # samples_test.append(inputs)
            out = get_module(self).pixelcnn(**inputs)
            # calc accuracy
            acc = (out[..., :-1].argmax(-1) == tgts).float().sum()
        if 'autoregressive' in preds_types:
            predictions = self.predict_autoregressive(encoder_input, src_mask, space_mask, tgt_space_mask,
                                                      shift_value, theta, seq_start_end, batch)
            prediction_dict['autoregressive'] = predictions

        out = {}
        for name, pred in prediction_dict.items():
            my_pred = pred.pop('pred')
            pred_abs = unrotate_and_unshift_batch(my_pred[..., :2], shift_value, theta, unrotate=False)
            if self.args.apply_normalization:
                pred_abs = denormalize_data(pred_abs, means, seq_start_end)
            if tgt is not None:
                loss = self.validation_loss(my_pred, tgt_norm, reduce='raw')
            else:
                loss = None
            out[name] = {'pred': my_pred, 'pred_abs': pred_abs, 'loss': loss}
            if 'teacher_forcing' in name:
                out[name].update({'acc': acc})
            out[name].update(pred)

        return out


def calc_perplexity(indices: torch.Tensor, k: int) -> float:
    min_encodings = torch.zeros(indices.shape[0], k).to(indices.device)
    min_encodings.scatter_(1, indices, 1)
    e_mean = torch.mean(min_encodings, dim=0)
    perp = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
    return perp.item()
