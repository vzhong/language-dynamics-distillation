from wrangl.learn import SupervisedModel
import torch
from torch import nn
from torch.nn import functional as F


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = nn.Embedding(256, cfg.demb)

        self.scene_trans = nn.Sequential(
            nn.Conv2d(2*cfg.demb, 256, (3, 3), 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, (3, 3), 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, (3, 3), 1, 1),
            nn.LeakyReLU(),
        )
        self.action_trans = nn.Sequential(
            nn.Linear(47*1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
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
        return feat['action'].view(-1).long()

    def forward(self, feat, batch):
        cur = self.emb(feat['features'].squeeze(1).squeeze(1).long())
        nxt = self.emb(feat['nxt_features'].squeeze(1).squeeze(1).long())
        scenes = torch.cat([cur, nxt-cur], dim=-1)
        scenes = scenes.transpose(1, 3)  # B, D, W, H
        conv = self.scene_trans(scenes)

        actions = []
        for i, xs in enumerate(feat['x'].squeeze(1).squeeze(1)):
            si = conv[i]
            actions.append(torch.stack([si[:, x] for x in xs.tolist()], dim=0))
        actions = torch.stack(actions, dim=0)

        valid = feat['valid'].squeeze(1).squeeze(1)

        B, A, D, H = actions.shape
        scores = self.action_trans(actions.view(B, A, -1))
        scores = scores.squeeze(-1) - (1-valid) * 1e20
        return scores

    def get_callbacks(self):
        return []
