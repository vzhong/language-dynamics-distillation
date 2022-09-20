from wrangl.learn import SupervisedModel
import torch
from torch import nn
import gym
import importlib
from torch.nn import functional as F
from argparse import Namespace


class Model(SupervisedModel):

    def __init__(self, cfg, orig_exp):
        super().__init__(cfg)
        self.orig_exp = orig_exp
        InnerModel = importlib.import_module('model.{}'.format(orig_exp['config']['model'])).Model
        from silg import envs
        self.env = gym.make(orig_exp['config']['env'])
        self.inner_model = InnerModel(Namespace(**orig_exp['config']), self.env)
        dconv = 64
        drep = self.orig_exp['config']['drep']
        if 'touchdown' in cfg.pretrain:
            self.proj = nn.Linear(dconv + drep, 30)  # number of class IDs in Touchdown segmentation maps
        else:
            self.proj = nn.Linear(dconv + drep, self.inner_model.name_emb.weight.size(1))

    def featurize(self, batch):
        # make it from B, 1, 1, *dims to 1, B, *dims
        feat = {}
        for k, v in batch.items():
            feat[k] = v.squeeze(1).squeeze(1).unsqueeze(0)
        feat['name'] = feat['name'].unsqueeze(-2)
        feat['next_name'] = feat['next_name'].squeeze(0).squeeze(-2).long()
        return feat

    def compute_loss(self, out, feat, batch):
        label = feat['next_name'].view(-1)
        logits = out.view(len(label), -1)
        return F.cross_entropy(logits, label, ignore_index=0)

    def compute_metrics(self, pred, gold) -> dict:
        valid = gold.ne(0).float()
        match = gold.eq(pred).float().mul(valid)
        acc = match.sum() / valid.sum()
        return dict(acc=acc.item())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['next_name']

    def forward(self, feat, batch):
        # add fake T
        inner_output, _ = self.inner_model.forward(feat, None, return_intermediate=True)
        conv = inner_output['conv']  # B, dconv, H, W
        rep = inner_output['rep']  # B, drep
        rep = rep.unsqueeze(2).unsqueeze(2).expand(rep.size(0), rep.size(1), conv.size(2), conv.size(3))
        cat = torch.cat([conv, rep], dim=1)  # B, dconv+drep, H, W
        cat = torch.permute(cat, (0, 3, 2, 1))  # B, W, H, dconv+drep
        proj = self.proj(cat)

        # shift right
        inp = feat['next_name'][:, :, :, :-1]
        pad = torch.ones(*feat['next_name'].shape[:-1]).unsqueeze(-1).to(inp.device).long()
        inp = torch.cat([pad, inp], dim=-1).squeeze()
        inp_emb = self.inner_model.emb(inp)
        proj_exp = proj.unsqueeze(-2).expand(proj.size(0), proj.size(1), proj.size(2), inp.size(-1), proj.size(3))

        cat = inp_emb + proj_exp
        B, H, W, max_seqlen, D = cat.size()
        out = torch.mm(cat.view(-1, cat.size(-1)), self.inner_model.emb.weight.t()).view(B, H, W, max_seqlen, -1)
        return out

    def get_callbacks(self):
        return []
