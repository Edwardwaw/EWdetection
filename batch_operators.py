
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np

from operators import BaseOperator
from op_helper import jaccard_overlap




class PadBatch(BaseOperator):
    """
    function : Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, sample, context=None):
        """
        Args:
            sample : a sample, which is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return sample
        max_shape = np.array(sample['image'].shape)

        if coarsest_stride > 0:
            max_shape[1] = int(np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)


        im = sample['image']
        im_c, im_h, im_w = im.shape[:]
        padding_im = np.zeros((im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        sample['image'] = padding_im
        if self.use_padded_im_info:
            sample['im_info'][:2] = max_shape[1:3]

        return sample




class RandomShape(BaseOperator):
    """
    Randomly reshape a batch.
    If random_inter is True, also randomly select one an interpolation algorithm from [cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4].
    If random_inter is False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, sample, context=None):
        # random select shape, interp method
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter else cv2.INTER_NEAREST

        im = sample['image']
        h, w = im.shape[:2]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        im = cv2.resize(im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
        sample['image'] = im
        if self.resize_box and 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] *scale_array, 0, float(shape) - 1)
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]

        return sample


class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.

    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadMultiScaleTest, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        for i in range(len(samples)):
            sample = samples[i]
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros((im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
        if not batch_input:
            samples = samples[0]
        return samples


