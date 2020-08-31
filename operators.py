try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import uuid
import warnings
import numpy as np
import random
import math
from numbers import Number
import cv2
from PIL import Image,ImageEnhance,ImageDraw

from op_helper import (filter_and_process,generate_sample_bbox,satisfy_sample_constraint,clip_bbox,data_anchor_sampling,
                       satisfy_sample_constraint_coverage,bbox_area_sampling,crop_image_sampling,generate_sample_bbox_square)



class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)



class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False, with_cutmix=False,with_mosaic=False):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
            with_cutmix (bool): whether or not to cutmix image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        self.with_cutmix = with_cutmix
        self.with_mosaic=with_mosaic
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im=sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode


        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # im.shape=[h,w,3]  3==>bgr
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            warnings.warn(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]

        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            warnings.warn(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array([im.shape[0], im.shape[1], 1.], dtype=np.float32)

        # 同样加载 mixup/cutup image xinx，为后面的mixup/cutup作准备
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)

        # decode cutmix image
        if self.with_cutmix and 'cutmix' in sample:
            self.__call__(sample['cutmix'], context)

        if self.with_mosaic and 'mosaic' in sample:
            for i in range(3):
                # print(i,' th image', sample['mosaic'][i]['im_file'])
                self.__call__(sample['mosaic'][i],context)

        return sample


########################################################################################################################
####  数据混合增强方案    mixup  cut mix  mosaic augmentation  ------------------------------------------------------------

class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        # mixup 以新图像左上角对对齐基准
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        # factor >= 1.0 or <= 0.0 ,  do nothings
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        # 仅在score(置信度)这个维度进行*factor这个操作
        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate((gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)

        is_crowd1 = sample['is_crowd']
        is_crowd2 = sample['mixup']['is_crowd']
        is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)

        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['is_crowd'] = is_crowd
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


class CutmixImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(CutmixImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _rand_bbox(self, img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        img_1 = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')
        img_2 = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2[:img2.shape[0], :img2.shape[1], :] = img2.astype('float32')
        img_1[bby1:bby2, bbx1:bbx2, :] = img_2[bby1:bby2, bbx1:bbx2, :]

        print('debug', bbx1, bby1, bbx2, bby2)
        print('debug', img1.shape, img2.shape)

        return img_1

    def __call__(self, sample, context=None):
        if 'cutmix' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('cutmix')
            return sample
        if factor <= 0.0:
            return sample['cutmix']
        img1 = sample['image']
        img2 = sample['cutmix']['image']
        img = self._rand_bbox(img1, img2, factor)

        # 代完成事项：此处gt box gt score gt class等的后处理方式纠正

        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['cutmix']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['cutmix']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = sample['gt_score']
        gt_score2 = sample['cutmix']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        sample['image'] = img
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]
        sample.pop('cutmix')
        return sample




class MosaicImage(BaseOperator):

    def __init__(self,half_mosaic_size=416):
        super(MosaicImage, self).__init__()
        self.half_mosaic_size=half_mosaic_size

    def __call__(self, sample,context=None):

        if 'mosaic' not in sample:
            return sample

        labels4=[]
        s=int(self.half_mosaic_size)
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

        # print(sample.keys())
        samples=[sample,sample['mosaic'][0],sample['mosaic'][1],sample['mosaic'][2]]

        for i,sam in enumerate(samples):
            img=sam['image']
            h,w=int(sam['h']),int(sam['w'])

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # print('debug for mosiac',y1a,y2a,x1a,x2a,y1b,y2b,x1b,x2b)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # boxes
            x = sam['gt_bbox']
            labels = x.copy()
            if x.size > 0:  # to pixel xyxy format
                labels[:, 0] += padw
                labels[:, 1] += padh
                labels[:, 2] += padw
                labels[:, 3] += padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            labels4=np.clip(labels4[:, :], 0, 2 * s)  # use with random_affine


        gt_bbox = labels4
        gt_class = np.concatenate([sam['gt_class'] for sam in samples], axis=0)
        gt_score = np.concatenate([sam['gt_score'] for sam in samples], axis=0)
        is_crowd = np.concatenate([sam['is_crowd'] for sam in samples], axis=0)
        difficult= np.concatenate([sam['difficult'] for sam in samples], axis=0)

        sample['image'] = img4
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['is_crowd'] = is_crowd
        sample['difficult'] = difficult
        sample['h'] = 2*s
        sample['w'] = 2*s
        sample.pop('mosaic')
        return sample




##----------------------------------------------------------------------------------------------------------------------

class Lighting(BaseOperator):
    """
    Lighting the imagen by eigenvalues and eigenvectors
    Args:
        eigval (list): eigenvalues
        eigvec (list): eigenvectors
        alphastd (float): random weight of lighting, 0.1 by default
    """
    def __init__(self, eigval, eigvec, alphastd=0.1):
        super(Lighting, self).__init__()
        self.alphastd = alphastd
        self.eigval = np.array(eigval).astype('float32')
        self.eigvec = np.array(eigvec).astype('float32')

    def __call__(self, sample, context=None):
        alpha = np.random.normal(scale=self.alphastd, size=(3, ))
        sample['image'] += np.dot(self.eigvec, self.eigval * alpha)
        return sample




class ColorDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings.
            in [lower, upper, probability] format.
        saturation (list): saturation settings.
            in [lower, upper, probability] format.
        contrast (list): contrast settings.
            in [lower, upper, probability] format.
        brightness (list): brightness settings.
            in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD) order.
        hsv_format (bool): whether to convert color from BGR to HSV
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 hsv_format=False,
                 random_channel=False):
        super(ColorDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.hsv_format = hsv_format
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 0] += random.uniform(low, high)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
            return img

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 1] *= delta
            return img
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness,
                self.apply_contrast,
                self.apply_saturation,
                self.apply_hue,
            ]
            distortions = np.random.permutation(functions)
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)

        if np.random.randint(0, 2):
            img = self.apply_contrast(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample





class RandomDistort(BaseOperator):
    def __init__(self,
                 brightness_lower=0.5,
                 brightness_upper=1.5,
                 contrast_lower=0.5,
                 contrast_upper=1.5,
                 saturation_lower=0.5,
                 saturation_upper=1.5,
                 hue_lower=-18,
                 hue_upper=18,
                 brightness_prob=0.5,
                 contrast_prob=0.5,
                 saturation_prob=0.5,
                 hue_prob=0.5,
                 count=4,
                 is_order=False):
        """
        Args:
            brightness_lower/ brightness_upper (float): the brightness
                between brightness_lower and brightness_upper
            contrast_lower/ contrast_upper (float): the contrast between
                contrast_lower and contrast_lower
            saturation_lower/ saturation_upper (float): the saturation
                between saturation_lower and saturation_upper
            hue_lower/ hue_upper (float): the hue between
                hue_lower and hue_upper
            brightness_prob (float): the probability of changing brightness
            contrast_prob (float): the probability of changing contrast
            saturation_prob (float): the probability of changing saturation
            hue_prob (float): the probability of changing hue
            count (int): the kinds of doing distrot
            is_order (bool): whether determine the order of distortion
        """
        super(RandomDistort, self).__init__()
        self.brightness_lower = brightness_lower
        self.brightness_upper = brightness_upper
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper
        self.hue_lower = hue_lower
        self.hue_upper = hue_upper
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.saturation_prob = saturation_prob
        self.hue_prob = hue_prob
        self.count = count
        self.is_order = is_order

    def random_brightness(self, img):
        brightness_delta = np.random.uniform(self.brightness_lower,
                                             self.brightness_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.brightness_prob:
            img = ImageEnhance.Brightness(img).enhance(brightness_delta)
        return img

    def random_contrast(self, img):
        contrast_delta = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.contrast_prob:
            img = ImageEnhance.Contrast(img).enhance(contrast_delta)
        return img

    def random_saturation(self, img):
        saturation_delta = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.saturation_prob:
            img = ImageEnhance.Color(img).enhance(saturation_delta)
        return img

    def random_hue(self, img):
        hue_delta = np.random.uniform(self.hue_lower, self.hue_upper)
        prob = np.random.uniform(0, 1)
        if prob < self.hue_prob:
            img = np.array(img.convert('HSV'))
            img[:, :, 0] = img[:, :, 0] + hue_delta
            img = Image.fromarray(img, mode='HSV').convert('RGB')
        return img

    def __call__(self, sample, context):
        """random distort the image"""
        ops = [
            self.random_brightness, self.random_contrast,
            self.random_saturation, self.random_hue
        ]
        if self.is_order:
            prob = np.random.uniform(0, 1)
            if prob < 0.5:
                ops = [
                    self.random_brightness,
                    self.random_saturation,
                    self.random_hue,
                    self.random_contrast,
                ]
        else:
            ops = random.sample(ops, self.count)
        assert 'image' in sample, "image data not found"
        im = sample['image']
        im = Image.fromarray(im)
        for id in range(self.count):
            im = ops[id](im)
        im = np.asarray(im)
        sample['image'] = im
        return sample



##----------------------------------------------------------------------------------------------------------------------
######### random expand     random crop   random flip ------------------------------------------------------------------


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
        is_mask_expand(bool): whether expand the segmentation.
    """

    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=(127.5, ) * 3):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        if isinstance(fill_value, Number):
            fill_value = (fill_value, ) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def __call__(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        height = int(sample['h'])
        width = int(sample['w'])

        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = img.astype(np.uint8)

        sample['h'] = h
        sample['w'] = w
        sample['image'] = canvas
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] += np.array([x, y] * 2, dtype=np.float32)
        return sample



class ExpandImage(BaseOperator):
    def __init__(self, max_ratio, prob, mean=[127.5, 127.5, 127.5]):
        """
        warning: gt_box should be normalized here.

        Args:
            max_ratio (float): the ratio of expanding
            prob (float): the probability of expanding image
            mean (list): the pixel mean
        """
        super(ExpandImage, self).__init__()
        self.max_ratio = max_ratio
        self.mean = mean
        self.prob = prob

    def __call__(self, sample, context):
        """
        Expand the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Construct new images with new height and width.
            3. Fill the new image with the mean.
            4. Put original imge into new image.
            5. Rescale the bounding box.
            6. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """

        prob = np.random.uniform(0, 1)
        assert 'image' in sample, 'not found image data'
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        if prob < self.prob:
            if self.max_ratio - 1 >= 0.01:
                expand_ratio = np.random.uniform(1, self.max_ratio)
                height = int(im_height * expand_ratio)
                width = int(im_width * expand_ratio)
                h_off = math.floor(np.random.uniform(0, height - im_height))
                w_off = math.floor(np.random.uniform(0, width - im_width))
                # expand_bbox: new box coords by old image
                expand_bbox = [
                    -w_off / im_width, -h_off / im_height,
                    (width - w_off) / im_width, (height - h_off) / im_height
                ]
                expand_im = np.ones((height, width, 3))
                expand_im = np.uint8(expand_im * np.squeeze(self.mean))
                expand_im = Image.fromarray(expand_im)
                im = Image.fromarray(im)
                expand_im.paste(im, (int(w_off), int(h_off)))
                expand_im = np.asarray(expand_im)
                gt_bbox, gt_class, _ = filter_and_process(expand_bbox,
                                                          gt_bbox, gt_class)
                sample['image'] = expand_im
                sample['gt_bbox'] = gt_bbox
                sample['gt_class'] = gt_class
                sample['w'] = width
                sample['h'] = height

        return sample




class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(max(min_ar, scale**2), min(max_ar, scale**-2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(gt_bbox, np.array([crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(sample['gt_score'], valid_ids, axis=0)
                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        # 将以image为坐标的gt box转为以crop image为坐标的gt box
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,centers < crop[2:]).all(axis=1)
        valid = np.logical_and(valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]



class CropImage(BaseOperator):
    def __init__(self, batch_sampler, satisfy_all=False, avoid_no_bbox=True):
        """
        warning: gt_box should be normalized here

        Args:
            batch_sampler (list): Multiple sets of different parameters for cropping.
            e.g.[[1,    1,   1.0,   1.0,   1.0,   1.0,   0.0,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.1,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.3,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.5,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.7,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.9,   1.0],
                 [1,   50,   0.3,   1.0,   0.5,   2.0,   0.0,   1.0]]
           [max sample, max trial, min scale, max scale, min aspect ratio, max aspect ratio, min overlap, max overlap]
            satisfy_all (bool): whether all boxes must satisfy.
            avoid_no_bbox (bool): whether to to avoid the
                                  situation where the box does not appear.
        """
        super(CropImage, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def __call__(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']

        gt_bbox = gt_bbox.tolist()
        sampled_bbox = []
        for sampler in self.batch_sampler:
            found = 0
            # 进行sampler[1]次采样尝试
            for i in range(sampler[1]):
                if found >= sampler[0]:
                    break
                sample_bbox = generate_sample_bbox(sampler)
                if satisfy_sample_constraint(sampler, sample_bbox, gt_bbox,self.satisfy_all):
                    sampled_bbox.append(sample_bbox)
                    found = found + 1
        im = np.array(im)
        while sampled_bbox:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            sample_bbox = sampled_bbox.pop(idx)
            sample_bbox = clip_bbox(sample_bbox)
            crop_bbox, crop_class, crop_score = filter_and_process(sample_bbox, gt_bbox, gt_class, scores=gt_score)
            if self.avoid_no_bbox:
                if len(crop_bbox) < 1:
                    continue
            xmin = int(sample_bbox[0] * im_width)
            xmax = int(sample_bbox[2] * im_width)
            ymin = int(sample_bbox[1] * im_height)
            ymax = int(sample_bbox[3] * im_height)
            im = im[ymin:ymax, xmin:xmax]
            sample['image'] = im
            sample['gt_bbox'] = crop_bbox
            sample['gt_class'] = crop_class
            sample['gt_score'] = crop_score
            #########################################
            # attention: 这里是不是应该加上下面的两行代码？
            # ???????
            sample['h'] = im.shape[0]
            sample['w'] = im.shape[1]
            return sample
        return sample



class CropImageWithDataAchorSampling(BaseOperator):
    def __init__(self,
                 batch_sampler,
                 anchor_sampler=None,
                 target_size=None,
                 das_anchor_scales=[16, 32, 64, 128],
                 sampling_prob=0.5,
                 min_size=8.,
                 avoid_no_bbox=True):
        """
        Args:
            anchor_sampler (list): anchor_sampling sets of different parameters for cropping.
            batch_sampler (list): Multiple sets of different parameters for cropping.
              e.g.[[1,   10,   1.0,   1.0,   1.0,   1.0,   0.0,   0.0,   0.2,   0.0]]
                  [[1,   50,   1.0,   1.0,   1.0,   1.0,   0.0,   0.0,   1.0,   0.0],
                   [1,   50,   0.3,   1.0,   1.0,   1.0,   0.0,   0.0,   1.0,   0.0],
                   [1,   50,   0.3,   1.0,   1.0,   1.0,   0.0,   0.0,   1.0,   0.0],
                   [1,   50,   0.3,   1.0,   1.0,   1.0,   0.0,   0.0,   1.0,   0.0],
                   [1,   50,   0.3,   1.0,   1.0,   1.0,   0.0,   0.0,   1.0,   0.0]]
              [max sample, max trial, min scale, max scale, min aspect ratio, max aspect ratio,
               min overlap, max overlap, min coverage, max coverage]
            target_size (bool): target image size.
            das_anchor_scales (list[float]): a list of anchor scales in data anchor sampling.
            min_size (float): minimum size of sampled bbox.
            avoid_no_bbox (bool): whether to to avoid the situation where the box does not appear.
        """
        super(CropImageWithDataAchorSampling, self).__init__()
        self.anchor_sampler = anchor_sampler
        self.batch_sampler = batch_sampler
        self.target_size = target_size
        self.sampling_prob = sampling_prob
        self.min_size = min_size
        self.avoid_no_bbox = avoid_no_bbox
        self.das_anchor_scales = np.array(das_anchor_scales)

    def __call__(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        image_width = sample['w']
        image_height = sample['h']
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']

        gt_bbox = gt_bbox.tolist()
        sampled_bbox = []

        prob = np.random.uniform(0., 1.)
        # anchor sampling
        if prob > self.sampling_prob:
            assert self.anchor_sampler
            for sampler in self.anchor_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = data_anchor_sampling(gt_bbox, image_width, image_height,
                                                        self.das_anchor_scales, self.target_size)
                    if sample_bbox == 0:
                        break
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)

                crop_bbox, crop_class, crop_score = filter_and_process(sample_bbox, gt_bbox, gt_class, scores=gt_score)
                crop_bbox, crop_class, crop_score = bbox_area_sampling(crop_bbox, crop_class, crop_score,
                                                                       self.target_size,self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                im = crop_image_sampling(im, sample_bbox, image_width, image_height, self.target_size)
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                sample['gt_score'] = crop_score
                #########################################
                # attention: 这里是不是应该加上下面的两行代码？
                # ???????
                sample['h'] = im.shape[0]
                sample['w'] = im.shape[1]
                return sample
            return sample
        # batch sampling
        else:
            for sampler in self.batch_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = generate_sample_bbox_square(sampler, image_width, image_height)
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)
                sample_bbox = clip_bbox(sample_bbox)

                crop_bbox, crop_class, crop_score = filter_and_process(
                    sample_bbox, gt_bbox, gt_class, scores=gt_score)
                # sampling bbox according the bbox area
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                xmin = int(sample_bbox[0] * image_width)
                xmax = int(sample_bbox[2] * image_width)
                ymin = int(sample_bbox[1] * image_height)
                ymax = int(sample_bbox[3] * image_height)
                im = im[ymin:ymax, xmin:xmax]
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                sample['gt_score'] = crop_score
                #########################################
                # attention: 这里是不是应该加上下面的两行代码？
                # ???????
                sample['h'] = im.shape[0]
                sample['w'] = im.shape[1]
                return sample
            return sample




class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.(Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                # 原有的gt box 横坐标
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample




class RandomScaledCrop(BaseOperator):
    """Resize image and bbox based on long side (with optional random scaling),
       then crop or pad image to target size.
    Args:
        target_dim (int): target size.
        scale_range (list): random scale range.
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self,
                 target_dim=512,
                 scale_range=[.1, 2.],
                 interp=cv2.INTER_LINEAR):
        super(RandomScaledCrop, self).__init__()
        self.target_dim = target_dim
        self.scale_range = scale_range
        self.interp = interp

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim   # output size of image
        random_dim = int(dim * random_scale)  # scaled/cropped image size
        dim_max = max(h, w)    # current image size

        scale = random_dim / dim_max
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            shift_array = np.array([offset_x, offset_y] * 2, dtype=np.float32)
            boxes = sample['gt_bbox'] * scale_array - shift_array
            boxes = np.clip(boxes, 0, dim - 1)
            # filter boxes with no area
            area = np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)
            valid = (area > 1.).nonzero()[0]
            sample['gt_bbox'] = boxes[valid]
            sample['gt_class'] = sample['gt_class'][valid]

        img = sample['image']
        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
            offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['h'] = dim
        sample['w'] = dim
        sample['image'] = canvas
        sample['im_info'] = [resize_h, resize_w, scale]
        return sample




class ResizeAndPad(BaseOperator):
    """Resize image and bbox, then pad image to target size.
    Args:
        target_dim (int): target size
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self, target_dim=512, interp=cv2.INTER_LINEAR):
        super(ResizeAndPad, self).__init__()
        self.target_dim = target_dim
        self.interp = interp

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']
        interp = self.interp
        dim = self.target_dim  # target output size
        dim_max = max(h, w)
        scale = self.target_dim / dim_max
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, dim - 1)
        img = sample['image']
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:resize_h, :resize_w, :] = img
        sample['h'] = dim
        sample['w'] = dim
        sample['image'] = canvas
        sample['im_info'] = [resize_h, resize_w, scale]
        return sample




class MultiscaleTestResize(BaseOperator):
    def __init__(self,
                 origin_target_size=800,
                 origin_max_size=1333,
                 target_size=[],
                 max_size=2000,
                 interp=cv2.INTER_LINEAR,
                 use_flip=True):
        """
        for multiscale test, only change image size but gt-bbox

        Rescale image to the each size in target size, and capped at max_size.
        Args:
            origin_target_size(int): original target size of image's short side.
            origin_max_size(int): original max size of image.
            target_size (list): A list of target sizes of image's short side.
            max_size (int): the max size of image.
            interp (int): the interpolation method.
            use_flip (bool): whether use flip augmentation.
        """
        super(MultiscaleTestResize, self).__init__()
        self.origin_target_size = int(origin_target_size)
        self.origin_max_size = int(origin_max_size)
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_flip = use_flip

        if not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid. Must be List, now is {}".format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.origin_target_size, int) and isinstance(
                self.origin_max_size, int) and isinstance(self.max_size, int)
                and isinstance(self.interp, int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy for multi-scale test.
        """
        origin_ims = {}  # consist by different images
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        base_name_list = ['image']
        origin_ims['image'] = im
        if self.use_flip:
            sample['image_flip'] = im[:, ::-1, :]
            base_name_list.append('image_flip')
            origin_ims['image_flip'] = sample['image_flip']

        for base_name in base_name_list:
            im_scale = float(self.origin_target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.origin_max_size:
                im_scale = float(self.origin_max_size) / float(im_size_max)

            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = np.round(im_scale_x * float(im_shape[1]))
            resize_h = np.round(im_scale_y * float(im_shape[0]))
            im_resize = cv2.resize(
                origin_ims[base_name],
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)

            sample[base_name] = im_resize
            info_name = 'im_info' if base_name == 'image' else 'im_info_image_flip'
            # sample[base_name] = im_resize
            sample[info_name] = np.array([resize_h, resize_w, im_scale], dtype=np.float32)

            for i, size in enumerate(self.target_size):
                im_scale = float(size) / float(im_size_min)
                if np.round(im_scale * im_size_max) > self.max_size:
                    im_scale = float(self.max_size) / float(im_size_max)
                im_scale_x = im_scale
                im_scale_y = im_scale
                resize_w = np.round(im_scale_x * float(im_shape[1]))
                resize_h = np.round(im_scale_y * float(im_shape[0]))
                im_resize = cv2.resize(
                    origin_ims[base_name],
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=self.interp)

                im_info = [resize_h, resize_w, im_scale]
                # hard-code here, must be consistent with
                # ppdet/modeling/architectures/input_helper.py
                name = base_name + '_scale_' + str(i)
                info_name = 'im_info_' + name
                sample[name] = im_resize
                sample[info_name] = np.array(
                    [resize_h, resize_w, im_scale], dtype=np.float32)
        return sample






###---------------------------------------------------------------------------------------------------------------------
######  random erase image    grid mask data augmention    auto augmentation  ------------------------------------------

class RandomErasingImage(BaseOperator):
    def __init__(self, prob=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        在图像分类中，按效果排序,random cropping > random flipping > random ereasing,三者联合使用会得更优的结果
        Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
        Args:
            prob (float): probability to carry out random erasing
            sl (float): lower limit of the erasing area ratio
            sh (float): upper limit of the erasing area ratio
            r1 (float): aspect ratio of the erasing region
        注意：此处box的实现与源论文不同，在某些gt box内部进行了random ereasing,不涉及gt box的改变
        """
        super(RandomErasingImage, self).__init__()
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))

            for idx in range(gt_bbox.shape[0]):
                if self.prob <= np.random.rand():
                    continue

                x1, y1, x2, y2 = gt_bbox[idx, :]
                w_bbox = x2 - x1 + 1
                h_bbox = y2 - y1 + 1
                area = w_bbox * h_bbox

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < w_bbox and h < h_bbox:
                    off_y1 = random.randint(0, int(h_bbox - h))
                    off_x1 = random.randint(0, int(w_bbox - w))
                    im[int(y1 + off_y1):int(y1 + off_y1 + h), int(x1 + off_x1):int(x1 + off_x1 + w), :] = 0
            sample['image'] = im

        sample = samples if batch_input else samples[0]
        return sample



class GridMaskOp(BaseOperator):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 current_iter=0,
                 upper_iter=360000):
        """
        GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086

        效果优于cutmix   auto aumentation

        warning: 1.每次迭代或者每个epoch要重新设置self.current_iter即要传入当前迭代次数


        Args:
            use_h (bool): whether to mask vertically
            use_w (bool): whether to mask horizontally
            rotate (float): angle for the mask to rotate
            offset (float): mask offset
            ratio (float): mask ratio
            mode (int): gridmask mode
            prob (float): max probability to carry out gridmask
            upper_iter (int): suggested to be equal to global max_iter
        """
        super(GridMaskOp, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.upper_iter = upper_iter
        self.current_iter=current_iter

        from gridmask_utils import GridMask
        self.gridmask_op = GridMask(
            use_h,
            use_w,
            rotate=rotate,
            offset=offset,
            ratio=ratio,
            mode=mode,
            prob=prob,
            upper_iter=upper_iter)

    def set_iter(self,current_iter):
        # attention: please reset self.current_iter every epoch or every iteration
        self.current_iter=current_iter

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            sample['image'] = self.gridmask_op(sample['image'],self.current_iter)
        sample = samples if batch_input else samples[0]
        return sample



class AutoAugmentImage(BaseOperator):
    def __init__(self, is_normalized=False, autoaug_type="v1"):
        """
        Args:
            is_normalized (bool): whether the bbox scale to [0,1]  (whether normalize gt-box)
            autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
        """
        super(AutoAugmentImage, self).__init__()
        self.is_normalized = is_normalized
        self.autoaug_type = autoaug_type
        if not isinstance(self.is_normalized, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """
        Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            if len(gt_bbox) == 0:
                continue

            # attention
            # gt_boxes : [x1, y1, x2, y2]
            # norm_gt_boxes: [y1, x1, y2, x2]
            height, width, _ = im.shape
            norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
            if not self.is_normalized:
                norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
                norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
                norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
                norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)
            else:
                norm_gt_bbox[:, 0] = gt_bbox[:, 1]
                norm_gt_bbox[:, 1] = gt_bbox[:, 0]
                norm_gt_bbox[:, 2] = gt_bbox[:, 3]
                norm_gt_bbox[:, 3] = gt_bbox[:, 2]

            from autoaugment_utils import distort_image_with_autoaugment
            im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,self.autoaug_type)

            if not self.is_normalized:
                gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
                gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
                gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
                gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)
            else:
                gt_bbox[:, 0] = norm_gt_bbox[:, 1]
                gt_bbox[:, 1] = norm_gt_bbox[:, 0]
                gt_bbox[:, 2] = norm_gt_bbox[:, 3]
                gt_bbox[:, 3] = norm_gt_bbox[:, 2]

            # gt_box=[x1,y1,x2,y2]
            sample['gt_bbox'] = gt_bbox
            sample['image'] = im

        sample = samples if batch_input else samples[0]
        return sample




########################################################################################################################
##----------------------------------------------------------------------------------------------------------------------

class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im

        sample = samples if batch_input else samples[0]
        return sample




class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox
        return sample




class Permute(BaseOperator):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    if self.channel_first:
                        im = np.swapaxes(im, 1, 2)
                        im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples




class Resize(BaseOperator):
    """Resize image and bbox.
    Args:
        target_dim (int or list): target size, can be a single number or a list (for random shape).
        interp (int or str): interpolation method, can be an integer or 'random' (for randomized interpolation).
                                default to `cv2.INTER_LINEAR`.
    """

    def __init__(self, target_dim=[], interp=cv2.INTER_LINEAR):
        super(Resize, self).__init__()
        self.target_dim = target_dim
        self.interp = interp  # 'random' for yolov3

    def __call__(self, sample, context=None):
        w = sample['w']
        h = sample['h']

        interp = self.interp
        if interp == 'random':
            interp = np.random.choice(range(5))
        # if target_dim is a sequence, randomly choice a target size
        if isinstance(self.target_dim, Sequence):
            dim = np.random.choice(self.target_dim)
        else:
            dim = self.target_dim
        resize_w = resize_h = dim
        scale_x = dim / w
        scale_y = dim / h
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, dim - 1)
        sample['scale_factor'] = [scale_x, scale_y] * 2
        sample['h'] = resize_h
        sample['w'] = resize_w

        sample['image'] = cv2.resize(sample['image'], (resize_w, resize_h), interpolation=interp)
        return sample






class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
        """
        Rescale image to the specified target size, and capped at max_size if max_size != 0.
        If target_size is list, selected a scale randomly as the specified target size.
        Args:
            target_size(int|list): the target size of image's short side, multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        return sample





class RandomInterpImage(BaseOperator):
    def __init__(self, target_size=0, max_size=0):
        """
        Random reisze image by multiply interpolate method.
        Args:
            target_size (int): the taregt size of image's short side
            max_size (int): the max size of image
        """
        super(RandomInterpImage, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        if not (isinstance(self.target_size, int) and isinstance(self.max_size, int)):
            raise TypeError('{}: input type is invalid.'.format(self))
        interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.resizers = []
        for interp in interps:
            self.resizers.append(ResizeImage(target_size, max_size, interp))

    def __call__(self, sample, context=None):
        """Resise the image numpy by random resizer."""
        resizer = random.choice(self.resizers)
        return resizer(sample, context)




class NormalizePermute(BaseOperator):
    """Normalize and permute channel order.
    Args:
        mean (list): mean values in RGB order.
        std (list): std values in RGB order.
    """

    def __init__(self,mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.120, 57.375]):
        super(NormalizePermute, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample, context=None):
        img = sample['image']
        img = img.astype(np.float32)

        img = img.transpose((2, 0, 1))
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        invstd = 1. / std
        for v, m, s in zip(img, mean, invstd):
            v.__isub__(m).__imul__(s)
        sample['image'] = img
        return sample




class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        super(PadBox, self).__init__()
        self.num_max_boxes = num_max_boxes

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes

        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)

        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)

            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample




class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """
    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample








########################################################################################################################
##----------------------------------------------------------------------------------------------------------------------


class ToTensor(BaseOperator):
    def __init__(self,fields=None):
        super(ToTensor, self).__init__()
        self.fields=fields

    def __call__(self, sample, context = None):
        import torch
        sample['image'] = torch.from_numpy(sample['image'])
        fields = self.fields if self.fields else []
        if 'gt_bbox' in fields:
            sample['gt_bbox'] = torch.from_numpy(sample['gt_bbox'])
        if 'gt_class' in fields:
            sample['gt_class']=torch.from_numpy(sample['gt_class'])
        if 'gt_score' in fields:
            sample['gt_score']=torch.from_numpy(sample['gt_score'])

        return sample
