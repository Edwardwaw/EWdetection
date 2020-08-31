import os
import numpy as np
import xml.etree.ElementTree as ET
import warnings
from torch.utils.data import Dataset
import copy
import traceback
import random
import collections



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





class VOCDataSet(Dataset):
    """
    Load dataset with PascalVOC format.

    VOC数据集中：如果在yaml配置文件中设置use_default_label=False，将从label_list.txt中读取类别列表，
    反之则可以没有label_list.txt文件，PaddleDetection会使用source/voc.py里的默认类别字典。
    label_list.txt每一行文本表示一个类别

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        use_default_label (bool): whether use the default mapping of label to integer index. Default True.
        with_background (bool): whether load background as a class, default True.
        label_list (str): if use_default_label is False, will load mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 split='train',
                 use_default_label=True,
                 with_background=True,
                 label_list='label_list.txt',
                 sample_transforms=None,
                 mixup=0.,
                 cutmix=0.,
                 mosaic=0.,
                 class_aware_sampling=False,
                 remove_empty=False,
                 inputs_def=None):
        super(VOCDataSet, self).__init__()

        # block 1: init dataset and load annotations
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.split=split
        self.with_background = with_background
        self.use_default_label = use_default_label

        self.cname2cid = None   # 'cname2id' is a dict to map category name to class id
        self._imid2path = None
        self.roidbs = None
        self.label_list = label_list
        self.load_roidb_and_cname2cid()  # load annotation info of all the dataset into a list of dict

        # sampling
        self.mixup=mixup
        self.cutmix=cutmix
        self.mosaic=mosaic
        self.class_aware_sampling=class_aware_sampling
        self.remove_empty=remove_empty

        self._load_img = False  # 是否加载图片
        self._sample_num = len(self.roidbs)  # 样本个数

        if self.class_aware_sampling:
            # warning: self.indices should be updated after one epoch training finishs
            self.img_weights = _calc_img_weights(self.roidbs)
            self.indices = random.choices(range(self._sample_num), weights=self.img_weights, k=self._sample_num)
        else:
            self.indices=[i for i in range(self._sample_num)]

        # transform
        '''
        inputs_def: # 网络输入的定义
            fields: ['image', 'gt_bbox', 'gt_class', 'gt_score']
        '''
        self._fields = copy.deepcopy(inputs_def[
            'fields']) if inputs_def else None  # input def
        self._sample_transforms = Compose(sample_transforms,{'fields': self._fields})



    def load_roidb_and_cname2cid(self):
        # load relative info into a list of dict
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        records = []
        ct = 0
        cname2cid = {}
        if not self.use_default_label:
            label_path = os.path.join(self.dataset_dir, self.label_list)
            if not os.path.exists(label_path):
                raise ValueError("label_list {} does not exists".format(label_path))
            with open(label_path, 'r') as fr:
                label_id = int(self.with_background)
                for line in fr.readlines():
                    cname2cid[line.strip()] = label_id
                    label_id += 1
        else:
            cname2cid = pascalvoc_label(self.with_background)

        image_sets_file = os.path.join(self.dataset_dir, "ImageSets", "Main", "%s.txt" % self.split)
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())

        for image_id in ids:
            xml_file = os.path.join(self.dataset_dir, "Annotations", "%s.xml" % image_id)
            img_file= os.path.join(self.dataset_dir, "JPEGImages", "%s.jpg" % image_id)
            if not os.path.exists(img_file):
                warnings.warn('Illegal image file: {}, and it will be ignored'.format(img_file))
                continue
            if not os.path.isfile(xml_file):
                warnings.warn('Illegal xml file: {}, and it will be ignored'.format(xml_file))
                continue
            tree = ET.parse(xml_file)

            #  image_id如果有则采用标注中的，如果没有则采用默认值
            if tree.find('id') is None:
                im_id = np.array([ct])
            else:
                im_id = np.array([int(tree.find('id').text)])

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)

            if im_w < 0 or im_h < 0:
                warnings.warn('Illegal width: {} or height: {} in annotation, '
                    'and {} will be ignored'.format(im_w, im_h, xml_file))
                continue
            gt_bbox = []
            gt_class = []
            gt_score = []
            is_crowd = []
            difficult = []
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                _difficult = int(obj.find('difficult').text)
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                if x2 > x1 and y2 > y1:
                    gt_bbox.append([x1, y1, x2, y2])
                    gt_class.append([cname2cid[cname]])
                    gt_score.append([1.])
                    is_crowd.append([0])
                    difficult.append([_difficult])
                else:
                    warnings.warn('Found an invalid bbox in annotations: xml_file: {}'
                        ', x1: {}, y1: {}, x2: {}, y2: {}.'.format(xml_file, x1, y1, x2, y2))
            gt_bbox = np.array(gt_bbox).astype('float32')    # shape=[n,4]
            gt_class = np.array(gt_class).astype('int32')    # shape=[n]
            gt_score = np.array(gt_score).astype('float32')  # shape=[n]
            is_crowd = np.array(is_crowd).astype('int32')    # shape=[n]
            difficult = np.array(difficult).astype('int32')  # shape=[n]
            voc_rec = {
                'im_file': img_file,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'difficult': difficult
            }
            if len(objs) != 0:
                records.append(voc_rec)

            ct += 1

        assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
        print('{} samples in file {}'.format(ct, anno_path))
        self.roidbs, self.cname2cid = records, cname2cid

    # --------------------------------------------------------------------------------------------------

    def __getitem__(self, idx):
        index=self.indices[idx]
        sample1=self.roidbs[index]

        sample = copy.deepcopy(sample1)

        if np.random.uniform(0,1) < self._load_img:
            sample['image'] = self._load_image(sample['im_file'])

        if np.random.uniform(0,1) < self.mixup:
            mix_idx = np.random.randint(1, self._sample_num)
            mix_index = self.indices[(mix_idx + idx - 1) % self._sample_num]
            sample['mixup'] = copy.deepcopy(self.roidbs[mix_index])
            # if self._load_img:
            #     sample['mixup']['image'] = self._load_image(sample['mixup']['im_file'])

        if np.random.uniform(0,1) < self.cutmix:
            mix_idx = np.random.randint(1, self._sample_num)
            mix_index = self.indices[(mix_idx + idx - 1) % self._sample_num]
            sample['cutmix'] = copy.deepcopy(self.roidbs[mix_index])
            # if self._load_img:
            #     sample['cutmix']['image'] = self._load_image(sample['cutmix']['im_file'])

        if np.random.uniform(0,1) < self.mosaic:
            mix_index = np.random.choice(self.indices, size=3, replace=False)
            sample['mosaic'] = copy.deepcopy([self.roidbs[id0] for id0 in mix_index])


        while _has_empty(sample['gt_bbox']) and self.remove_empty:
            idx+=1
            idx%=self._sample_num
            index = self.indices[idx]
            sample = copy.deepcopy(self.roidbs[index])

        # sample=self._sample_transforms(sample)
        # return collect_needed_item(sample,self._fields)   # return a list
        # why deepcopy sample? 否则sample这个量会保存在self.roidbs的内存空间中不会释放

        sample = self._sample_transforms(sample)
        return collect_needed_item(sample, self._fields)  # return a list


    def __len__(self):
        return len(self.roidbs)


    ## help/load function -------------------------------------------------------------------------------

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

    def _load_image(self, filename):
        with open(filename, 'rb') as f:
            return f.read()






def pascalvoc_label(with_background=True):
    labels_map = {
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
