import time
import torch
import numpy as np
from hackrl.wrappers import _tile_characters_to_image, _initialize_char_array


class ImageConverter:

    def __init__(self, font_size=9, rescale_font_size=(6, 6)):
        char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = char_array.shape[2]
        self.char_width = char_array.shape[3]
        self.char_array = char_array.transpose(0, 1, 4, 2, 3)

    def convert(self, obs, crop_size=12, blstats_cursor=False):
        char_array = self.char_array
        char_height = self.char_height
        char_width = self.char_width
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
