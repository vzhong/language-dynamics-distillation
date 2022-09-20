import torch
import numpy as np
from nle.env import tasks as nle_tasks


class NetHackChallengeMessages(nle_tasks.NetHackChallenge):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_len = 16
        self.empty = np.ones(80, dtype=np.uint8) * 32
        self.context = [self.empty] * self.context_len
        self.next_message = self.empty

    def reset(self, *args, **kwargs):
        self.context = [self.empty] * self.context_len
        return super().reset(*args, **kwargs)

    def _get_observation(self, observation):
        ret = super()._get_observation(observation)
        tty_chars_idx = self._observation_keys.index('tty_chars')
        chars = observation[tty_chars_idx]  # according to nle NetHackChallenge in tasks.py
        msg = chars[0].copy()

        self.context.append(self.next_message)
        self.context.pop(0)
        self.next_message = msg

        ret['msg_context'] = np.stack(self.context)
        return ret
