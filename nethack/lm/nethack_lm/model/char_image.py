from .char import Model as Base
import torch
from transformers.modeling_outputs import BaseModelOutput


class Model(Base):

    def __init__(self, hackrl_model):
        super().__init__(hackrl_model)
        self.screen_encoder = hackrl_model.screen_encoder

    def get_inputs(self, inputs, ns=None, autodevice=True):
        ns = super().get_inputs(inputs, ns=ns, autodevice=autodevice)
        ns.screen_image = inputs['screen_image']
        return ns

    def get_forward_args(self, ns):
        T, B, C, H, W = ns.screen_image.shape
        topline = ns.tty_chars[..., 0, :]
        top = self.topline_encoder(
            topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        bottom_line = ns.tty_chars[..., -2:, :]
        bottom = self.bottomline_encoder(
            bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
        )

        screen = self.screen_encoder(
            ns.screen_image
            .float(memory_format=torch.contiguous_format)
            .view(T * B, C, H, W)
        )

        st = torch.cat([bottom, top, screen], dim=1).view(T, B, -1)
        core, _ = self.core(st)
        return dict(encoder_outputs=BaseModelOutput(last_hidden_state=core.transpose(0, 1)))
