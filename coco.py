import os
import numpy as np
from torch.utils.data import Dataset
import warnings
import copy
import traceback
import random
import collections


########################################################################################################################
########################################################################################################################

class Compose(object):
    def __init__(self, transforms, ctx=None):
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                warnings.warn("fail to map op [{}] with error: {} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data



def _calc_img_weights(roidbs):
    """
    calculate the probabilities of each sample
    """
    imgs_cls = []
    num_per_cls = {}
    img_weights = []
    # 统计每个类别的gt-box的数量
    for i, roidb in enumerate(roidbs):
        img_cls = set([k for cls in roidbs[i]['gt_class'] for k in cls])
        imgs_cls.append(img_cls)
        for c in img_cls:
            if c not in num_per_cls:
                num_per_cls[c] = 1
            else:
                num_per_cls[c] += 1

    for i in range(len(roidbs)):
        weights = 0
        for c in imgs_cls[i]:
            weights += 1 / num_per_cls[c]
        img_weights.append(weights)
    # probabilities sum to 1
    img_weights = img_weights / np.sum(img_weights)
    return img_weights



def _has_empty(item):
    # check whether the item is empty
    def empty(x):
        if isinstance(x, np.ndarray) and x.size == 0:
            return True
        elif isinstance(x, collections.Sequence) and len(x) == 0:
            return True
        else:
            return False

    if isinstance(item, collections.Sequence) and len(item) == 0:
        return True
    if item is None:
        return True
    if empty(item):
        return True
    return False



def collect_needed_item(samples, fields):
    # just keep needed key-value in sample, return a tuple
    def im_shape(samples, dim=3):
        # hard code
        assert 'h' in samples
        assert 'w' in samples
        if dim == 3:  # RCNN, ..  ==> return np.array((h,w,1))
            return np.array((samples['h'], samples['w'], 1), dtype=np.float32)
        else:  # YOLOv3, ..   ==> return np.array((h,w))
            return np.array((samples['h'], samples['w']), dtype=np.int32)

    one_ins = ()
    for i, field in enumerate(fields):
        if field == 'im_shape':
            one_ins += (im_shape(samples), )
        elif field == 'im_size':
            one_ins += (im_shape(samples, 2), )
        else:
            if field == 'is_difficult':
                field = 'difficult'
            assert field in samples, '{} not in samples'.format(field)
            one_ins += (samples[field], )

    # one_ins = {}
    # for i, field in enumerate(fields):
    #     if field == 'im_shape':
    #         one_ins[field]=im_shape(samples)
    #     elif field == 'im_size':
    #         one_ins[field]=im_shape(samples, 2)
    #     else:
    #         if field == 'is_difficult':
    #             field = 'difficult'
    #         assert field in samples, '{} not in samples'.format(field)
    #         one_ins[field]=samples[field]
    return one_ins


########################################################################################################################
########################################################################################################################



class COCODataSet(Dataset):
    """
    Load COCO records with annotations in json file 'anno_path'

    `roidbs` is list of dict whose structure is:
    {
        'im_file': im_fname, # image file name
        'im_id': img_id, # image id
        'h': im_h, # height of image
        'w': im_w, # width
        'is_crowd': is_crowd,
        'gt_score': gt_score,
        'gt_class': gt_class,
        'gt_bbox': gt_bbox,
        'gt_poly': gt_poly,
    }

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): json file path.
        sample_num (int): number of samples to load, -1 means all.
        with_background (bool): whether load background as a class.
            if True, total class number will be 81. default True.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 with_background=True,
                 sample_transforms=None,
                 mixup=0.,
                 cutmix=0.,
                 mosaic=0.,
                 class_aware_sampling=False,
                 remove_empty=False,
                 inputs_def=None):
        super(COCODataSet, self).__init__()

        # block 1: init dataset and load annotations
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.with_background = with_background

        self.roidbs = None # roidb is a list of dict, which storage annotation info
        self.cname2cid = None  # 'cname2id' is a dict to map category name to class id
        self._imid2path = None

        self.load_roidb_and_cname2cid()    # load annotation info of all the dataset into a list of dict

        # block 2: sample settings
        self.mixup = mixup
        self.cutmix = cutmix
        self.mosaic = mosaic
        self.class_aware_sampling = class_aware_sampling
        self.remove_empty = remove_empty

        self._sample_num = len(self.roidbs)  # 样本个数

        if self.class_aware_sampling:
            # warning: self.indices should be updated after one epoch training finishs
            self.img_weights = _calc_img_weights(self.roidbs)
            self.indices = random.choices(range(self._sample_num), weights=self.img_weights, k=self._sample_num)
        else:
            self.indices=[i for i in range(self._sample_num)]

        # block 3: transform
        '''
                inputs_def: # 网络输入的定义
                    fields: ['image', 'gt_bbox', 'gt_class', 'gt_score']
                '''
        self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None  # input def
        self._sample_transforms = Compose(sample_transforms, {'fields': self._fields})



    def load_roidb_and_cname2cid(self):
        # 读取所有数据的信息，每张图片的相关信息存放在一个字典上，整个数据集的信息存放在元素为字典的列表里
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)
        assert anno_path.endswith('.json'),'invalid coco annotation file: ' + anno_path

        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        # 记录标注数据
        records = []
        ct = 0

        # when with_background = True, mapping category to classid, like:
        #   background:0, first_class:1, second_class:2, ...
        # cat id==>cls id  (前者是由数据集决定，后者采用默认值)
        catid2clsid = dict({
            catid: i + int(self.with_background)
            for i, catid in enumerate(cat_ids)
        })
        # cat/class name==>cls id  (前者是由数据集决定，后者采用默认值)
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })



        for img_id in img_ids:
            img_anno = coco.loadImgs(img_id)[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,im_fname) if image_dir else im_fname
            if not os.path.exists(im_path):
                warnings.warn('Illegal image file: {}, and it will be '
                            'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                warnings.warn('Illegal width: {} or height: {} in annotation, '
                            'and im_id: {} will be ignored'.format(im_w, im_h,img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            }


            ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                x, y, box_w, box_h = inst['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    inst['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(inst)
                else:
                    warnings.warn(
                        'Found an invalid bbox in annotations: im_id: {}, '
                        'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_id, float(inst['area']), x1, y1, x2, y2))
            num_bbox = len(bboxes)

            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.ones((num_bbox, 1), dtype=np.float32)
            is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)

            for i, box in enumerate(bboxes):
                catid = box['category_id']
                gt_class[i][0] = catid2clsid[catid]
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i][0] = box['iscrowd']


            coco_rec.update({
                'difficult':difficult,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
            })

            # print('Load file: {}, im_id: {}, h: {}, w: {}.'.format(im_path, img_id, im_h, im_w))
            records.append(coco_rec)
            ct += 1
        assert len(records) > 0, 'not found any coco record in %s' % (anno_path)
        print('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid

    ## ------------------------------------------------------------------------------------------

    def __getitem__(self, idx):
        index = self.indices[idx]
        sample1 = self.roidbs[index]

        sample = copy.deepcopy(sample1)

        if np.random.uniform(0,1) < self.mixup:
            mix_idx = np.random.randint(1, self._sample_num)
            mix_index = self.indices[(mix_idx + idx - 1) % self._sample_num]
            sample['mixup'] = copy.deepcopy(self.roidbs[mix_index])

        if np.random.uniform(0,1) < self.cutmix:
            mix_idx = np.random.randint(1, self._sample_num)
            mix_index = self.indices[(mix_idx + idx - 1) % self._sample_num]
            sample['cutmix'] = copy.deepcopy(self.roidbs[mix_index])

        if np.random.uniform(0,1) < self.mosaic:
            mix_index=np.random.choice(self.indices,size=3,replace=False)
            # print(mix_index)
            sample['mosaic'] = copy.deepcopy([self.roidbs[id0] for id0 in mix_index])


        while _has_empty(sample['gt_bbox']) and self.remove_empty:
            idx+=1
            idx%=self._sample_num
            index = self.indices[idx]
            # sample = self.roidbs[index]
            sample=copy.deepcopy(self.roidbs[index])

        # sample_temp = copy.deepcopy(sample)
        sample = self._sample_transforms(sample)
        return collect_needed_item(sample, self._fields)  # return a list


    def __len__(self):
        return len(self.roidbs)

    ### help/load function -----------------------------------------------------------------------
    def get_roidb(self):
        #获取数据信息列表 list[dict], len=num of images, 每个数据的信息用一个dict保存
        return self.roidbs

    def get_cname2cid(self):
        #获取标签ID(dict)类别名到id的映射
        return self.cname2cid

    def get_anno(self):
        # 获取标注文件路径
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)

    def get_imid2path(self):
        # 获取图片路径   return dict, 图片路径
        return self._imid2path



'''
self.roidbs=xxx_rec(list len=有标注的图片数量,其中的每个元素是一个如下所示的字典。) 
xxx_rec = {
    'im_file': im_fname,         # 一张图像的完整路径
    'im_id': np.array([img_id]), # 一张图像的ID序号
    'h': im_h,                   # 图像高度
    'w': im_w,                   # 图像宽度
    'is_crowd': is_crowd,        # 是否是群落对象, 默认为0 (VOC中无此字段，为了使个别模型同时适配于COCO与VOC)
    'gt_class': gt_class,        # 标注框标签名称的ID序号
    'gt_bbox': gt_bbox,          # 标注框坐标(xmin, ymin, xmax, ymax)
    'gt_score': gt_score,        # 标注框置信度得分 (此字段为了适配Mixup操作)
    'gt_poly': gt_poly,          # 分割掩码，此字段只在coco_rec中出现，默认为None
    'difficult': difficult       # 是否是困难样本，此字段只在voc_rec中出现，默认为0
}

'''
