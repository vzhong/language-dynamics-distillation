import torch
from wrangl.learn import SupervisedModel
from torch import nn
from torch.nn import functional as F
from messenger.models.emma import EMMA, nonzero_mean


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        model_kwargs = {
            "hist_len": 3,
            "n_latent_var": 128,
            "emb_dim": 256,
        }
        self.emma = EMMA(**model_kwargs)
        self.out = nn.Sequential(
            nn.Linear(5184, 10*10*3*128),
        )

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean().item())

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out.view(-1, 128), feat['next_state'].view(-1))

    def extract_context(self, feat, batch):
        return ['n/a'] * len(batch)

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['next_state']

    def forward(self, feat, batch):
        state = feat['curr_state']  # 3, 10, 10, 2
        temb = feat['text']  # 3, 36, 768

        out = []
        for state_i, emb_i in zip(state, temb):
            out.append(self.one_step(state_i, emb_i))
        out = torch.stack(out, dim=0)
        return out

    def one_step(self, state, temb):
        emma = self.emma
        # split the state tensor into objects and avatar
        obj_state, avt_state = torch.split(state, [state.shape[-1]-1, 1], dim=-1)

        # embedding for the avatar object, which will not attend to text
        avatar_emb = nonzero_mean(emma.sprite_emb(avt_state))

        # take the non_zero mean of embedded objects, which will act as attention query
        query = nonzero_mean(emma.sprite_emb(obj_state))

        # Attention
        key = emma.txt_key(temb)
        key_scale = emma.scale_key(temb)  # (num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1)

        value = emma.txt_val(temb)
        val_scale = emma.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=1)

        state_emb, weights = emma.attention(query, key, value)

        # compress the channels from KHWC to HWC' where K is history length
        state_emb = state_emb.view(emma.state_h, emma.state_w, -1)
        avatar_emb = avatar_emb.view(emma.state_h, emma.state_w, -1)

        # Take the average between state_emb and avatar_emb in case of overlap
        state_emb = (state_emb + avatar_emb) / 2.0

        # permute from HWC to NCHW and do convolution
        state_emb = state_emb.permute(2, 0, 1).unsqueeze(0)
        state_emb = F.leaky_relu(emma.conv(state_emb)).view(-1)

        # sprite_emb = emma.sprite_emb(state[-1]).view(10, 10, -1)
        return self.out(state_emb).view(10, 10, 3, 128)
