from wrangl.learn import SupervisedModel
import torch
from torch import nn
from torch.nn import functional as F


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = nn.Embedding(cfg.nwords, cfg.demb)

        self.wiki_rnn = nn.LSTM(cfg.demb, cfg.drnn, batch_first=True, bidirectional=True)
        self.task_rnn = nn.LSTM(cfg.demb, cfg.drnn, batch_first=True, bidirectional=True)
        self.inv_rnn = nn.LSTM(cfg.demb, cfg.drnn, batch_first=True, bidirectional=True)
        dname = 6 * 6
        self.out = nn.Sequential(
            nn.Linear(2*dname*cfg.demb + 2*cfg.drnn, 512),
            nn.Tanh(),
            nn.Linear(512, 5),  # 5 actions for RTFM
        )

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['action'].view(-1))

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['action'].view(-1)

    def forward(self, feat, batch):
        cur_name = self.emb(feat['name'].squeeze(1).squeeze(1).squeeze(-2))
        nxt_name = self.emb(feat['nxt_name'].squeeze(1).squeeze(1).squeeze(-2))
        name = torch.cat([cur_name, nxt_name-cur_name], dim=-1)

        name_emb = name.sum(-2)  # B, H, W, D
        name_emb = name_emb.view(name_emb.size(0), -1)

        wiki, _ = self.wiki_rnn(self.emb(feat['wiki'].squeeze(1).squeeze(1)))
        inv, _ = self.inv_rnn(self.emb(feat['inv'].squeeze(1).squeeze(1)))
        task, _ = self.task_rnn(self.emb(feat['task'].squeeze(1).squeeze(1)))

        wiki = wiki.sum(1)
        inv = inv.sum(1)
        task = task.sum(1)

        cat = torch.cat([name_emb, wiki + inv + task], dim=-1)
        out = self.out(cat)
        valid = feat['valid'].squeeze(1).squeeze(1)
        out = out - (1-valid) * 1e20
        return out

    def get_callbacks(self):
        return []
