import torch
from wrangl.learn import SupervisedModel
from torch import nn
from torch.nn import functional as F


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = nn.Embedding(512, 30)
        self.out = nn.Sequential(
            nn.Linear(48648, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 5),
        )

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean().item())

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out.view(-1, out.size(-1)), feat['action'].view(-1))

    def extract_context(self, feat, batch):
        return ['n/a'] * len(batch)

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['action']

    def forward(self, feat, batch):
        B = feat['curr_state'].size(0)

        flat_curr = self.emb(feat['curr_state'][:, -1]).view(B, -1)
        flat_text = feat['text'][:, -1].view(B, -1)  # only take last timestep
        flat_next = self.emb(feat['next_state']).view(B, -1)
        flat = torch.cat([flat_curr, flat_next, flat_text], dim=1)
        out = self.out(flat)
        return out
