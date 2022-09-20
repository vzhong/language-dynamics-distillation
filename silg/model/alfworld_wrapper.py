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
        self.inner_model = InnerModel(Namespace(**orig_exp['config']), self.env)
        drep = self.orig_exp['config']['drep']
        self.proj = nn.Linear(400 + drep, 100)
        self.text_decoder = nn.LSTM(200, 100, batch_first=True, num_layers=1)
        self.proj2 = nn.Linear(100 + 100, self.inner_model.name_emb.weight.size(1))

    def featurize(self, batch):
        # make it from B, 1, 1, *dims to 1, B, *dims
        feat = {}
        for k, v in batch.items():
            feat[k] = v.squeeze(1).squeeze(1).unsqueeze(0)
        feat['next_text'] = feat['next_text'].squeeze(0)
        return feat

    def compute_loss(self, out, feat, batch):
        label = feat['next_text']
        B, T, D = out.size()
        loss = F.cross_entropy(out.view(B*T, -1), label.view(-1))
        return loss

        label = feat['next_text']
        error = label - out
        squared_error = error * error
        mean_squared_error = torch.mean(squared_error)
        return mean_squared_error

    def compute_metrics(self, pred, gold) -> dict:
        match = pred.eq(gold).float()
        return dict(acc=match.mean().item())

    def extract_context(self, feat, batch):
        return 'n/a'

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['next_text']

    def forward(self, feat, batch):
        # add fake T
        inner_output, _ = self.inner_model.forward(feat, None, return_intermediate=True)
        rep = inner_output['rep']  # B, drep

        command = feat['command'].long()
        command_len = feat['command_len'].long()
        T, B, num_cmd, max_cmd_len = command.size()

        command_rnn = self.inner_model.run_rnn(self.inner_model.cmd_rnn, command.view(-1, max_cmd_len), command_len.view(-1))
        command_trans = self.inner_model.cmd_trans(command_rnn)

        # rep is (T*B, drep)
        rep_exp = rep.unsqueeze(1).expand(rep.size(0), num_cmd, rep.size(1))

        attn, _ = self.inner_model.run_attn(command_trans, command_len.reshape(-1), rep_exp.reshape(-1, rep.size(1)))

        attn_summ = attn.view(B, num_cmd, -1).sum(1)
        flat = torch.cat([attn_summ, rep], 1)

        proj = self.proj(flat).tanh()

        # shift label right
        inp = feat['next_text']
        inp = torch.cat([torch.zeros(inp.size(0), 1).long().to(inp.device), inp[:, :-1]], dim=1)
        inp_emb = self.inner_model.name_emb(inp)

        proj_exp = proj.unsqueeze(1).expand(inp.size(0), inp.size(1), proj.size(1))
        cat = torch.cat([inp_emb, proj_exp], dim=2)

        rnn, _ = self.text_decoder(cat)
        cat2 = torch.cat([rnn, proj_exp], dim=2)
        out = self.proj2(cat2).tanh()

        B, T, D = out.size()
        scores = out.view(-1, D).mm(self.inner_model.name_emb.weight.t()).view(B, T, -1)
        return scores

    def get_callbacks(self):
        return []
