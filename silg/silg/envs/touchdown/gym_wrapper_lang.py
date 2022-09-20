from pathlib import Path

import torch
import gym
import os
from gym.envs import registration
from collections import Counter

from . import gym_wrapper


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PARENT = Path(__file__).parent

TD_ROOT = os.environ.get('TOUCHDOWN_ROOT', os.path.join(ROOT, 'cache', 'touchdown'))
# register gym environments if not already registered
for split in ("train", "dev", "test"):
    if f"td_res50_lang_{split}-v0" not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=f"td_res50_lang_{split}-v0",
            entry_point="silg.envs.touchdown.gym_wrapper_lang:TDWrapperLang",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'pca_10.npz'),
                data_json=str(PARENT.joinpath(f"data/{split}.json")),
                feat_type='res50',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )

    if f"td_segs_lang_{split}-v0" not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=f"td_segs_lang_{split}-v0",
            entry_point="silg.envs.touchdown.gym_wrapper_lang:TDWrapperLang",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'maj_ds_a10.npz'),
                data_json=str(PARENT.joinpath(f"data/{split}.json")),
                feat_type='segs',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )


class TDWrapperLang(gym_wrapper.TDWrapper):

    def get_text_fields(self):
        return ['text', 'desc']

    def get_observation_space(self):
        obs = super().get_observation_space()
        obs.update({
            "desc": (self.max_text, ),
            "desc_len": (1, ),
        })
        return obs

    def convert_to_str(self, obs):
        ''' Reformat the obs from TDNavigator. WARNING: We implicitly assume _reformat() is called once
        during reset() and once during step()
        '''
        str_obs = super().convert_to_str(obs)
        x = obs['cur_x'].item()

        left_bound = x - 10
        right_bound = x + 10
        text_desc = []
        ignore = {gym_wrapper.CLASSES.index(k) for k in ['sky']}

        np_feats = obs['features'].numpy()
        field = np_feats[:, left_bound:right_bound].flatten().tolist()
        items = []
        for i, c in Counter(field).most_common(10):
            if i in ignore:
                continue
            n = c / len(field)
            if n > 0.1:
                adj = 'a lot of'
            elif n:
                adj = 'some'
            items.append('{} {}'.format(adj, gym_wrapper.CLASSES[i]))
        text_desc.append('In front of you, you see {}'.format(', '.join(items)))

        if left_bound > 0:
            field = np_feats[:, :left_bound].flatten().tolist()
            items = []
            for i, c in Counter(field).most_common(10):
                if i in ignore:
                    continue
                n = c / len(field)
                if n > 0.1:
                    adj = 'a lot of'
                elif n:
                    adj = 'some'
                items.append('{} {}'.format(adj, gym_wrapper.CLASSES[i]))
        text_desc.append('To your left, you see {}'.format(', '.join(items)))

        if right_bound < np_feats.shape[1]:
            field = np_feats[:, right_bound:].flatten().tolist()
            items = []
            for i, c in Counter(field).most_common(10):
                if i in ignore:
                    continue
                n = c / len(field)
                if n > 0.1:
                    adj = 'a lot of'
                elif n:
                    adj = 'some'
                items.append('{} {}'.format(adj, gym_wrapper.CLASSES[i]))
        text_desc.append('To your right, you see {}'.format(', '.join(items)))
        str_obs['desc'] = '. '.join(text_desc)
        return str_obs
