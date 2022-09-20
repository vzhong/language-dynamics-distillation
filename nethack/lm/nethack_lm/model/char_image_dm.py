from .char_image import Model as Base
import torch
import random
from torch.nn import functional as F
from torch import nn


class Model(Base):

    def __init__(self, hackrl_model):
        super().__init__(hackrl_model)
        self.n_vocab = 256
        self.proj = nn.Linear(512, 24*80*self.n_vocab)
        del self.lm

    def predict(self, inputs, prev_core):
        screen_image = inputs['screen_image'].unsqueeze(0)  # insert fake time dimension
        tty_chars = inputs['tty_chars'].unsqueeze(0)

        T, B, C, H, W = screen_image.shape
        topline = tty_chars[..., 0, :]
        top = self.topline_encoder(
            topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        bottom_line = tty_chars[..., -2:, :]
        bottom = self.bottomline_encoder(
            bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        screen = self.screen_encoder(
            screen_image
            .float(memory_format=torch.contiguous_format)
            .view(T * B, C, H, W)
        )

        st = torch.cat([bottom, top, screen], dim=1).view(T, B, -1)
        core, _ = self.core(st, prev_core)

        out = self.proj(core.transpose(0, 1)[:, -1])
        pred = out.view(-1, 24, 80, self.n_vocab).max(-1)[1]
        return pred

    def get_inputs(self, inputs, ns=None, autodevice=True):
        ns = super().get_inputs(inputs, ns=ns, autodevice=autodevice)
        ns.next_screen = inputs['next_tty_chars']
        return ns

    def training_step(self, batch, batch_idx, compute_loss=True):
        ns = self.get_inputs(batch)
        forward_args = self.get_forward_args(ns)
        out = self.proj(forward_args['encoder_outputs'].last_hidden_state[:, -1])
        gold = ns.next_screen

        loss = F.cross_entropy(out.view(-1, self.n_vocab), gold.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, split_name='val'):
        ns = self.get_inputs(batch)
        forward_args = self.get_forward_args(ns)
        out = self.proj(forward_args['encoder_outputs'].last_hidden_state[:, -1])
        pred = out.view(-1, 24, 80, self.n_vocab).max(-1)[1]
        gold = ns.next_screen

        acc = pred.eq(gold).float().mean().item()
        for c, p, g in zip(ns.context_strs, pred, gold):
            g = '\n'.join([row.tobytes().decode(errors='ignore') for row in g.cpu().numpy()])
            p = '\n'.join([row.tobytes().decode(errors='ignore') for row in p.cpu().numpy()])
            if random.random() < self.val_cache_p_record:
                self.val_cache.insert(0, (c, p, g))
        self.val_cache = self.val_cache[:self.max_val_cache]
        self.log('{}_acc'.format(split_name), acc)

    def get_early_stop_metric(self):
        return 'val_acc', 'max'
