#!/usr/bin/env python
import ray
import hydra
import torch
import logging
import warnings
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


class MyDataset(IterableDataset):

    def iterate_unprocessed(self):
        for raw in super().iterate_unprocessed():
            text = raw['text']
            trajs = raw['traj']
            for st, nt in trajs:
                nt = torch.from_numpy(nt['entities']).long()
                yield dict(text=text, curr_state=st, next_state=nt)


@hydra.main(config_path='dynamics_conf', config_name='default')
def main(cfg):
    Model = SupervisedModel.load_model_class(cfg.model)
    train = torch.load('saves/emma-1to2_train_rollouts.pt', map_location=torch.device('cpu'))
    val = torch.load('saves/emma-1to2_val_rollouts.pt', map_location=torch.device('cpu'))
    pool = ray.util.ActorPool([MyProcessor.remote() for _ in range(3)])
    train = MyDataset(train, pool)
    val = MyDataset(val, pool)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
