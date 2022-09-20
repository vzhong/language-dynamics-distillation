import os
import torch
import tqdm
import logging
import ray
import socket

import coolname
import hydra
import omegaconf

from concurrent import futures

import numpy as np
from nhtools import dataset as dt
from torch.utils.data import DataLoader
from hackrl.wrappers import _tile_characters_to_image, _initialize_char_array


@ray.remote
class ImageConverter:

    def convert_obs_to_img(self, obs, font_size=9, crop_size=12, rescale_font_size=(6, 6), blstats_cursor=False):
        char_array = _initialize_char_array(font_size, rescale_font_size)
        char_height = char_array.shape[2]
        char_width = char_array.shape[3]
        # Transpose for CHW
        char_array = char_array.transpose(0, 1, 4, 2, 3)

        half_crop_size = crop_size // 2
        output_height_chars = crop_size
        output_width_chars = crop_size
        chw_image_shape = (
            3,
            output_height_chars * char_height,
            output_width_chars * char_width,
        )

        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if crop_size:
            # Center around player
            if blstats_cursor:
                center_x, center_y = obs["blstats"][:2]
            else:
                center_y, center_x = obs["tty_cursor"]
            offset_h = center_y - half_crop_size
            offset_w = center_x - half_crop_size

        out_image = np.zeros(chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=output_height_chars,
            output_width_chars=output_width_chars,
            char_array=char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )
        return out_image


class NethackHuman2RLDataset(dt.TtyrecDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.converters = ray.util.ActorPool([ImageConverter.remote() for i in range(FLAGS.lm.num_actors)])

    def __iter__(self):
        for obs in super().__iter__():
            obs['next_tty_chars'] = obs['tty_chars'][:, -1]

            for k in ['tty_chars', 'tty_colors', 'tty_cursors', 'done']:
                obs[k] = obs[k].transpose(0, 1)  # make T, B
                obs[k] = obs[k][:-1]  # the last one is for next message
                if k == 'tty_cursors':  # make naming consistent
                    obs['tty_cursor'] = obs[k]
                    del obs[k]

            # T, B, H, W = obs['tty_chars'].size()
            # inputs = [dict(tty_chars=char, tty_colors=color, tty_cursor=cursor) for char, color, cursor in zip(obs['tty_chars'].reshape(T*B, H, W).numpy(), obs['tty_colors'].reshape(T*B, H, W).numpy(), obs['tty_cursor'].reshape(T*B, -1).numpy())]
            # images = self.converters.map(lambda actor, arg: actor.convert_obs_to_img.remote(arg), inputs)
            # x = np.stack(images, axis=0)
            # obs['screen_image'] = x.reshape(T, B, *x.shape[1:])
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
@hydra.main(config_path="confs", config_name="baseline_lm")
def main(cfg):
    global FLAGS
    FLAGS = cfg

    # ray.init()

    with futures.ThreadPoolExecutor(max_workers=FLAGS.lm.db_workers) as tp:
        train, valid = get_train_valid_test_split(batch_size=FLAGS.lm.batch_size, threadpool=tp)

        total = 100000
        data = []
        saves = []
        for i, batch in enumerate(tqdm.tqdm(train, total=total, desc='dumping data')):
            data.append(batch)
            if len(data) >= 10000:
                fsave = 'batches.{}.pt'.format(len(saves))
                saves.append(fsave)
                print('saving to {}'.format(fsave))
                torch.save(data, fsave)
                data.clear()
            if i >= total:
                break
        fsave = 'batches.{}.pt'.format(len(saves))
        saves.append(fsave)
        torch.save(data, fsave)
        data.clear()

        logging.info("Graceful exit. Bye bye!")


if __name__ == "__main__":
    main()
