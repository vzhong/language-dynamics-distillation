from wrangl.learn import SupervisedModel
import torch
from torch import nn
from torch.nn import functional as F


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = nn.Embedding(40000, cfg.demb)
        self.grid_rnn = nn.LSTM(2*cfg.demb, 256, batch_first=True, bidirectional=True)
        self.text_rnn = nn.LSTM(cfg.demb, 256, batch_first=True, bidirectional=True)
        self.cmd_rnn = nn.LSTM(cfg.demb, 256, batch_first=True, bidirectional=True)

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['action'].view(-1))

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['action'].view(-1).long()

    def forward(self, feat, batch):
        cur = self.emb(feat['name'].squeeze(1).squeeze(1).squeeze(-2))  # B, H, W, L, D
        nxt = self.emb(feat['nxt_name'].squeeze(1).squeeze(1).squeeze(-2))
        grid = torch.cat([cur, nxt-cur], dim=-1)
        B, H, W, L, demb = grid.shape
        grid_summ = self.grid_rnn(grid.view(-1, L, demb))[0].sum(1).view(B, H, W, -1)  # B, H, W, 2*drnn

        text = self.emb(feat['text'].squeeze(1).squeeze(1))  # B, T, D
        text_summ = self.text_rnn(text)[0].sum(1)  # B, 2*drnn

        cmd = self.emb(feat['command'].squeeze(1).squeeze(1))  # B, K, L
        B, num_cmds, cmd_len, demb = cmd.shape
        cmd_flat = self.cmd_rnn(cmd.view(B*num_cmds, cmd_len, demb))[0]
        cmd_summ = cmd_flat.sum(1).view(B, num_cmds, -1)  # B, num_cmds, 2*drnn

        grid_text_summ = text_summ.unsqueeze(1).unsqueeze(1).expand_as(grid_summ).mul(grid_summ).mean(1).mean(1)  # B, 2*drnn

        scores = grid_text_summ.unsqueeze(1).expand_as(cmd_summ).mul(cmd_summ).mean(2)
        valid = feat['valid'].squeeze(1).squeeze(1).float()
        norm = scores - (1-valid) * 1e20
        return norm

    def get_callbacks(self):
        return []
