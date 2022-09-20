from wrangl.learn import SupervisedModel
import torch
from torch.nn import functional as F
import gym
import importlib
from argparse import Namespace


class Model(SupervisedModel):

    def __init__(self, cfg, orig_exp, inverse_model=None):
        super().__init__(cfg)
        self.orig_exp = orig_exp
        self.inverse_model = inverse_model
        InnerModel = importlib.import_module('model.{}'.format(orig_exp['config']['model'])).Model
        from silg import envs
        self.env = gym.make(orig_exp['config']['env'])
        self.inner_model = InnerModel(Namespace(**orig_exp['config']), self.env)

    def featurize(self, batch):
        # make it from B, 1, 1, *dims to 1, B, *dims
        feat = {}
        for k, v in batch.items():
            feat[k] = v.squeeze(1).squeeze(1).unsqueeze(0)
        feat['actions'] = self.inverse_model.extract_pred(self.inverse_model(batch, batch), feat, batch)
        return feat

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out, feat['actions'])

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['actions']

    def forward(self, feat, batch):
        # add fake T
        inner_output, _ = self.inner_model.forward(feat, None)
        return inner_output['policy_logits'].squeeze(0)

    def get_callbacks(self):
        return []
