import os
import getpass
import logging
import pprint
import socket
import importlib

import coolname
import hydra
import omegaconf
import hackrl.models

from concurrent import futures

import numpy as np
from nhtools import dataset as dt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nethack_lm.image_converter import ImageConverter


class NethackHuman2RLDataset(dt.TtyrecDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if FLAGS.lm.process_image:
            self.converter = ImageConverter()

    def __iter__(self):
        for obs in super().__iter__():
            obs['next_tty_chars'] = x = obs['tty_chars'][:, -1]

            if FLAGS.lm.ignore_empty_target and (x != 32).sum() == 0:
                continue  # this has no message

            for k in ['tty_chars', 'tty_colors', 'tty_cursors', 'done']:
                obs[k] = obs[k].transpose(0, 1)  # make T, B
                obs[k] = obs[k][:-1]  # the last one is for next message
                if k == 'tty_cursors':  # make naming consistent
                    obs['tty_cursor'] = obs[k]
                    del obs[k]

            if FLAGS.lm.process_image:
                T, B, H, W = obs['tty_chars'].size()
                inputs = [dict(tty_chars=char, tty_colors=color, tty_cursor=cursor) for char, color, cursor in zip(obs['tty_chars'].reshape(T*B, H, W).numpy(), obs['tty_colors'].reshape(T*B, H, W).numpy(), obs['tty_cursor'].reshape(T*B, 2).numpy())]
                images = [self.converter.convert(o) for o in inputs]
                x = np.stack(images, axis=0)
                obs['screen_image'] = x.reshape(T, B, *x.shape[1:])
            yield obs


def uid():
    return "%s:%i:%s" % (socket.gethostname(), os.getpid(), coolname.generate_slug(2))


omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)


def get_train_valid_test_split(batch_size=1, threadpool=None):
    # TODO: Eventually we should change our dataset to move from selecting
    # rowids in the ttyrecs table, to rowids of a (yet to be created) user table
    dbfilename = FLAGS.lm.dbfilename

    bucket = 0
    num_buckets = 1

    sql_subset = (
        "SELECT games.gameid, ttyrecs.path, games.death, games.deathlev, games.points, ttyrecs.frames FROM ttyrecs "
        "INNER JOIN games ON ttyrecs.gameid=games.gameid "
        # "WHERE ttyrecs.is_clean_bl=1 AND games.death!='quit' "
        "WHERE games.death!='quit' "
        # "WHERE games.death='ascended' "
    )
    cond = []
    if cond:
        cond = " AND ((" + " AND ".join(cond) + ") OR games.death='ascended') "
        sql_subset = sql_subset + cond
    # Splitting the data into train and tests is not as simple as it looks,
    # since you can't easily query in SQLITE for users < "c" for instance (alphabetical sort)
    # For a fair split we NEED to separate the users.
    # TO DO: Fetch all the data and have a separate class that does the splitting for you after
    # FOR NOW: 683663 is our magic dividing line (user with one game) < 683663 ~ 1/4 of the data
    split = 683663
    train_dataset = NethackHuman2RLDataset(
        batch_size=batch_size,
        seq_length=FLAGS.unroll_length+1,
        dbfilename=dbfilename,
        threadpool=threadpool,
        sql_subset=sql_subset + f" AND games.gameid > {split}" + (" LIMIT 10000" if FLAGS.lm.debug else ""),
        shuffle=True,
    )
    train_dataset._rowids = train_dataset._rowids[bucket::num_buckets]

    # valid_start = train_end
    # valid_end = train_end + flags.validation_ttyrecs
    valid_dataset = NethackHuman2RLDataset(
        batch_size=batch_size,
        seq_length=FLAGS.unroll_length+1,
        dbfilename=dbfilename,
        threadpool=threadpool,
        sql_subset=sql_subset + f" AND games.gameid < {split}" + (" LIMIT 10000" if FLAGS.lm.debug else ""),
        shuffle=True,
    )
    valid_dataset._rowids = valid_dataset._rowids[bucket::num_buckets]

    return [DataLoader(dataset, batch_size=1, num_workers=0) for dataset in [train_dataset, valid_dataset]]


# Override config_path via --config_path.
@hydra.main(config_path="confs", config_name="lm")
def main(cfg):
    global FLAGS
    FLAGS = cfg

    if not os.path.isabs(FLAGS.savedir):
        FLAGS.savedir = os.path.join(hydra.utils.get_original_cwd(), FLAGS.savedir)

    logging.info("flags:\n%s\n", pprint.pformat(dict(FLAGS)))

    train_id = "%s/%s/%s" % (
        FLAGS.entity if FLAGS.entity is not None else getpass.getuser(),
        FLAGS.project,
        FLAGS.group,
    )

    logging.info("train_id: %s", train_id)

    hackrl_model = hackrl.models.create_model(FLAGS, FLAGS.device)
    Model = importlib.import_module('nethack_lm.model.{}'.format(FLAGS.lm.model)).Model
    model = Model(hackrl_model)

    model_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of model parameters: %i", model_numel)

    logger = None
    if FLAGS.wandb.enable:
        logger = WandbLogger(project=FLAGS.wandb.project, name=FLAGS.wandb.name, entity=FLAGS.wandb.entity)

    with futures.ThreadPoolExecutor(max_workers=FLAGS.lm.db_workers) as tp:
        train, valid = get_train_valid_test_split(batch_size=FLAGS.lm.batch_size, threadpool=tp)

        early_stop_metric, early_stop_mode = model.get_early_stop_metric()
        checkpoint = ModelCheckpoint(
            dirpath=FLAGS.savedir,
            monitor=early_stop_metric,
            mode=early_stop_mode,
            filename='{epoch:02d}-{' + early_stop_metric + ':.4f}',
            every_n_epochs=1, save_top_k=3,
            verbose=True, save_last=True,
        )

        flast = os.path.join(FLAGS.savedir, 'last.ckpt')
        if FLAGS.lm.autoresume and os.path.isfile(flast):
            FLAGS.lm.fresume = flast

        trainer = pl.Trainer(
            gpus=FLAGS.lm.gpus,
            max_steps=FLAGS.lm.num_train_steps,
            val_check_interval=FLAGS.lm.eval_period,
            limit_val_batches=FLAGS.lm.num_val_steps,
            auto_lr_find=False,
            resume_from_checkpoint=FLAGS.lm.fresume,
            log_every_n_steps=FLAGS.lm.log_period,
            callbacks=[checkpoint] + model.get_callbacks(),
            logger=logger,
            default_root_dir=FLAGS.savedir,
            weights_save_path=FLAGS.savedir,
        )

        if not FLAGS.lm.test_only:
            trainer.fit(model, train, valid)
        print(trainer.test(model, valid, verbose=True))

        logging.info("Graceful exit. Bye bye!")


if __name__ == "__main__":
    main()
