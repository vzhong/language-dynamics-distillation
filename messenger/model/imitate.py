import torch
from wrangl.learn import SupervisedModel
from torch.nn import functional as F
from model.reverse import Model as Reverse
from messenger.models.emma import EMMA, nonzero_mean


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.reverse = Reverse.load_from_checkpoint(cfg.freverse_checkpoint)
        model_kwargs = {
            "hist_len": 3,
            "n_latent_var": 128,
            "emb_dim": 256,
        }
        self.emma = EMMA(**model_kwargs)

    def compute_metrics(self, pred, gold) -> dict:
        return dict(acc=pred.eq(gold).float().mean().item())

    def compute_loss(self, out, feat, batch):
        return F.cross_entropy(out.view(-1, out.size(-1)), feat['action'].view(-1))

    def extract_context(self, feat, batch):
        return ['n/a'] * len(batch)

    def featurize(self, batch):
        feat = super().featurize(batch)
        with torch.no_grad():
            rev = self.reverse.extract_pred(self.reverse(feat, batch), feat, batch)
            feat['action'] = rev
        return feat

    def extract_pred(self, out, feat, batch):
        return out.max(-1)[1]

    def extract_gold(self, feat, batch):
        return feat['action']

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

        action_probs = emma.action_layer(state_emb)
        return action_probs
