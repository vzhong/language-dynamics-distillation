#!/usr/bin/env python
import os
import ray
import json
import hydra
import torch
import logging
import warnings
from hydra.utils import get_original_cwd
from wrangl.learn import SupervisedModel
from wrangl.data import IterableDataset, Processor


warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


logger = logging.getLogger(__name__)


@ray.remote
class MyProcessor(Processor):

    def process(self, raw):
        return raw


def generate_dataset(rollouts, seed=0, T=4):
    pool = ray.util.ActorPool([MyProcessor.remote()])
    flat = []
    for traj in rollouts:
        for i in range(len(traj)-T):
            keys = traj[i].keys()
            cur = {k: torch.cat([traj[t][k] for t in range(i, i+T)], dim=1) for k in keys}
            nxt = traj[i+T]
            if 'command' in nxt:
                cur['next_text'] = nxt['text']
            elif 'name' in nxt:
                cur['next_name'] = nxt['name']
            elif 'features' in nxt:
                pad_height, pad_width = 47, 10
                pad = torch.zeros(1, 1, pad_height, pad_width)
                padded_feat = torch.cat([pad, nxt['features'], pad], dim=-1)
                view = padded_feat[:, :, :, nxt['cur_x'].item():nxt['cur_x'].item()+2*pad_width+1]
                cur['next_view'] = view
            else:
                raise NotImplementedError()
            flat.append(cur)
    # if renorm:
    #     ys = torch.stack([x['next_count'] for x in flat], dim=0).float()
    #     mean = torch.mean(ys, dim=0)
    #     std = torch.clamp(torch.std(ys, dim=0), min=1e-5)
    #     for x in flat:
    #         x['next_count'] = (x['next_count'] - mean) / std
    return IterableDataset(flat, pool)


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    # load configs
    fexp = os.path.join(cfg.drollout, 'exp.json')
    print('loading config from {}'.format(fexp))
    with open(fexp) as f:
        exp = json.load(f)

    frollouts = os.path.join(cfg.drollout, 'rollout.pt')

    model_name = 'dynamics_wrapper'
    if 'alfworld' in cfg.pretrain:
        model_name = 'alfworld_wrapper'
        frollouts = os.path.join(get_original_cwd, 'alfworld_rollouts.pt')
    elif 'touchdown_lang' in cfg.pretrain:
        model_name = 'touchdown_wrapper'
        frollouts = os.path.join(get_original_cwd(), 'touchdown_lang_rollouts.pt')
    elif 'touchdown' in cfg.pretrain:
        model_name = 'touchdown_wrapper'
        frollouts = os.path.join(get_original_cwd(), 'touchdown_rollouts.pt')
    print('loading model {}'.format(model_name))
    Model = SupervisedModel.load_model_class(model_name)

    print('loading rollouts from {}'.format(frollouts))
    rollouts = torch.load(frollouts)
    num_train = int(0.85 * len(rollouts))
    train_rollouts = rollouts[:num_train]
    val_rollouts = rollouts[num_train:]
    print('loaded {} rollouts, using {} for train, {} for eval'.format(len(rollouts), len(train_rollouts), len(val_rollouts)))

    train = generate_dataset(train_rollouts)
    val = generate_dataset(val_rollouts)
    Model.run_train_test(cfg, train, val, model_kwargs=dict(orig_exp=exp))


if __name__ == '__main__':
    main()
