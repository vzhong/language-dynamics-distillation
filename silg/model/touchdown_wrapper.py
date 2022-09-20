from wrangl.learn import SupervisedModel
import torch
from torch import nn
from torch.nn import functional as F
import gym
import importlib
from argparse import Namespace


class Model(SupervisedModel):

    def __init__(self, cfg, orig_exp):
        super().__init__(cfg)
        self.orig_exp = orig_exp
        InnerModel = importlib.import_module('model.{}'.format(orig_exp['config']['model'])).Model
        from silg import envs
        self.env = gym.make(orig_exp['config']['env'])

        # assert not orig_exp['config']['stateful']
        orig_exp['config']['stateful'] = True

        inner_args = Namespace(**orig_exp['config'])
        self.inner_model = InnerModel(inner_args, self.env)

        drep = self.orig_exp['config']['drep']
        dconv = 64
        self.proj = nn.Sequential(
            # nn.Linear(drep + self.env.max_actions * dconv, 256),
            nn.Linear(drep, 256),
            nn.Tanh(),
            nn.Linear(256, 21*47*self.inner_model.id_emb.weight.size(0)),  # width * height * number of class IDs in Touchdown segmentation maps
        )

    def featurize(self, batch):
        # make it from B, 1, T, *dims to T, B, *dims
        feat = {}
        for k, v in batch.items():
            feat[k] = v.squeeze(1).transpose(0, 1).contiguous()
        feat['next_view'] = feat['next_view'].squeeze(1).squeeze(0)
        return feat

    def compute_loss(self, out, feat, batch):
        label = feat['next_view'].long()
        loss = F.cross_entropy(out.view(-1, out.shape[-1]), label.view(-1))
        return loss  # scale

    def compute_metrics(self, pred, gold) -> dict:
        acc = pred.max(-1)[1].eq(gold).float().mean()
        return dict(acc=acc.item())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out

    def extract_gold(self, feat, batch):
        return feat['next_view']

    def forward(self, feat, batch):
        # add fake T
        T = feat['features'].shape[0]
        B = feat['features'].shape[1]
        initial_state = [t.squeeze(0) for t in self.inner_model.initial_state(batch_size=B)]  # get rid of T dim
        inner_output, core_state = self.inner_model.forward(feat, initial_state, return_intermediate=True)

        # take the last time step
        rep = inner_output['rep'].reshape(T, B, -1)[-1]  # B, drep
        # conv = inner_output['conv']

        # pool = conv.max(3)[0].transpose(1, 2)  # TB, W, channels
        # slices = []
        # for x_i, feat_i in zip(feat['x'].view(T*B, -1), pool):
        #     slices_i = torch.index_select(feat_i, 0, x_i)  # select width channel
        #     slices.append(slices_i)
        # commands = torch.stack(slices, dim=0)  # TB, num_actions, channels
        # commands = commands.reshape(T, B, -1)[-1]
        # cat = torch.cat([rep, commands], -1)

        cat = rep
        out = self.proj(cat).view(B, 47, 21, -1)
        return out

    def get_callbacks(self):
        return []
