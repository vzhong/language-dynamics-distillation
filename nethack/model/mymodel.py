# Copyright (c) Meta Platforms, Inc. and its affiliates.
# by Eric Hambros
import torch
from torch import nn
import torch.nn.functional as F

from wrangl.learn.rl import MoolibVtrace


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


class ScreenEncoder(nn.Module):

    def __init__(self, dout=512, h=72, w=72):
        super().__init__()
        conv_layers = []
        self.h, self.w = h, w
        self.hidden_dim = dout

        self.conv_filters = [[3, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        for in_channels, out_channels, filter_size, stride in self.conv_filters:
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, filter_size, stride=stride)
            )
            conv_layers.append(nn.ELU(inplace=True))

            self.h = conv_outdim(self.h, filter_size, padding=0, stride=stride)
            self.w = conv_outdim(self.w, filter_size, padding=0, stride=stride)

        self.conv_head = nn.Sequential(*conv_layers)
        self.out_size = self.h * self.w * out_channels

        self.fc_head = nn.Sequential(
            nn.Linear(self.out_size, self.hidden_dim), nn.ELU(inplace=True)
        )

    def forward(self, screen_image):
        x = self.conv_head(screen_image / 255.0)
        x = x.view(-1, self.out_size)
        x = self.fc_head(x)
        return x


class MessageEncoder(nn.Module):

    def __init__(self, dout=128):
        super().__init__()
        from nle import nethack
        self.hidden_dim = dout
        i_dim = nethack.NLE_TERM_CO

        self.msg_fwd = nn.Sequential(
            nn.Linear(i_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = (message - 32) / 96.0
        return self.msg_fwd(message_normed)


class BLStatsEncoder(nn.Module):

    def __init__(self, dout=128):
        super().__init__()
        from nle import nethack

        self.conv_layers = []
        w = nethack.NLE_TERM_CO * 2
        for in_ch, out_ch, filt, stride in [[2, 32, 8, 4], [32, 64, 4, 1]]:
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, filt, stride=stride))
            self.conv_layers.append(nn.ELU(inplace=True))
            w = conv_outdim(w, filt, padding=0, stride=stride)

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fwd_net = nn.Sequential(
            nn.Linear(w * out_ch, 128),
            nn.ELU(),
            nn.Linear(128, dout),
            nn.ELU(),
        )
        self.hidden_dim = dout

    def forward(self, bottom_lines):
        B, D = bottom_lines.shape

        # ASCII 32: ' ', ASCII [33-128]: visible characters
        chars_normalised = (bottom_lines - 32) / 96

        # ASCII [45-57]: -./01234556789
        numbers_mask = (bottom_lines > 44) * (bottom_lines < 58)
        digits_normalised = numbers_mask * (bottom_lines - 47) / 10

        # Put in different channels & conv (B, 2, D)
        x = torch.stack([chars_normalised, digits_normalised], dim=1)
        return self.fwd_net(self.conv_net(x).view(B, -1))


class Model(MoolibVtrace):

    def __init__(self, cfg):
        super().__init__(cfg)
        from nle import nethack
        self.num_actions = len(nethack.ACTIONS)
        self.message_encoder = MessageEncoder()
        self.blstats_encoder = BLStatsEncoder()
        self.screen_encoder = ScreenEncoder()
        self.state_tracker = self.make_state_tracker(self.message_encoder.hidden_dim + self.blstats_encoder.hidden_dim + self.screen_encoder.hidden_dim, cfg.dstate)
        self.policy = nn.Linear(cfg.dstate, self.num_actions)
        self.baseline = nn.Linear(cfg.dstate, 1)

    def forward(self, inputs, state_tracker_state=None):
        T, B, C, H, W = inputs["screen_image"].shape

        message = self.message_encoder(inputs["tty_chars"][..., 0, :].float(memory_format=torch.contiguous_format).view(T * B, -1))
        blstats = self.blstats_encoder(inputs["tty_chars"][..., -2:, :].float(memory_format=torch.contiguous_format).view(T * B, -1))
        screen = self.screen_encoder(inputs["screen_image"].float(memory_format=torch.contiguous_format).view(T * B, -1, H, W))
        rep = torch.cat([screen, message, blstats], dim=1)
        state_rep, state_tracker_state = self.update_state(rep, state_tracker_state, done=inputs['done'])

        policy_logits = self.policy(state_rep)
        baseline = self.baseline(state_rep)
        action = torch.multinomial(F.softmax(policy_logits.reshape(T*B, -1), dim=1), num_samples=1)
        output = dict(
            policy_logits=policy_logits.view(T, B, -1),
            baseline=baseline.view(T, B),
            action=action.view(T, B),
        )
        return output, state_tracker_state
