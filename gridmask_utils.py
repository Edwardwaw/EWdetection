import numpy as np
from PIL import Image


class GridMask(object):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        '''
        param
        use_h:
        use_w:
        rotate:
        offset:
        ratio:
        mode: mode=1  ratio代表保留率， mode=0 ratio代表抛弃比率
        prob:
        upper_iter:
        '''
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.st_prob = prob
        self.upper_iter = upper_iter


    def __call__(self, x, curr_iter):
        self.prob = self.st_prob * min(1, 1.0 * curr_iter / self.upper_iter)
        if np.random.rand() > self.prob:
            return x
        h, w , _ = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        # l d
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        # 离图像左上角的偏置量
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        # 纵向设置
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        # 水平设置
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w].astype(np.float32)
        # print('enter grid mask op')
        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask, axis=-1)
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            x = (x * mask + offset * (1 - mask)).astype(x.dtype)
        else:
            x = (x * mask).astype(x.dtype)

        return x