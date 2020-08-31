from voc import *
from coco import *
from operators import *
from batch_operators import *
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from samplers import DistributedSampler,IterationBasedBatchSampler


def collate_fn(batch, fields=['image', 'gt_bbox', 'gt_class', 'gt_score']):
    '''
    Args
    batch(a list of tuple): len=batch_size  len(tuple)=num of elements returned by dataset
    :return:
    '''
    batch_data = list(zip(*batch))  # batch_data (a list of tuple)
    collate_dict = {fields[i]: i for i in range(len(fields))}
    return_list = []

    for key, value in collate_dict.items():
        if key == 'image':
            return_list.append(torch.stack(batch_data[value], 0))
        elif isinstance(key, str) and isinstance(batch_data[value][0], torch.Tensor):
            return_list.append(torch.cat(batch_data[value], 0))
        else:
            return_list.append(batch_data[value])
    # for item in return_list:
    #     if isinstance(item, torch.Tensor):
    #         print(item.shape)
    #     else:
    #         print(item)

    return return_list



trans=[DecodeImage(to_rgb=True,with_mixup=False,with_mosaic=True),
       # MixupImage(alpha=1.5,beta=1.5),
       MosaicImage(),
       ColorDistort(),
       RandomExpand(fill_value=[123.675,116.28,103.53]),
       RandomCrop(),
       RandomFlipImage(is_normalized=False),
       NormalizeBox(),
       # PadBox(num_max_boxes=50),
       BboxXYXY2XYWH(),
       RandomShape(sizes=[512],random_inter=True),

       NormalizeImage(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],is_scale=True,is_channel_first=False),
       Permute(to_bgr=False,channel_first=True),
       ToTensor(['gt_bbox', 'gt_class', 'gt_score'])
       ]



# voc_dataset = VOCDataSet(dataset_dir='VOC2007',
#                          image_dir='JPEGImages',
#                          anno_path='Annotations',
#                          split='test',
#                          use_default_label=True,
#                          with_background=False,
#                          label_list='label_list.txt',
#                          sample_transforms=trans ,
#                          mixup=0.5,
#                          cutmix=0.,
#                          mosaic=0.,
#                          class_aware_sampling=False ,
#                          remove_empty=True,
#                          inputs_def={'fields':['image', 'gt_bbox', 'gt_class', 'gt_score']})


coco_dataset=COCODataSet(dataset_dir='coco2017',
                         image_dir='val2017',
                         anno_path='annotations/instances_val2017.json',
                         with_background=False,
                         class_aware_sampling=False,
                         mixup=0.0,
                         cutmix=0.0,
                         mosaic=0.5,
                         remove_empty=True,
                         sample_transforms=trans,
                         inputs_def={'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']})


def make_dataloader(dataset, batch_size=8, num_workers=8, is_train=True, distributed=False,
                     max_iter=None, start_iter=0):
    shuffle = is_train or distributed
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    # attention:  batch sampler reyturn a list of indicies belong to one batch
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    if max_iter is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter,start_iter=start_iter)

    data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler,
                             pin_memory=False, collate_fn=collate_fn)

    return data_loader


dataloader=make_dataloader(coco_dataset,distributed=False)

for i,data in enumerate(dataloader):
    print('the ',i,'st batch input data, its statistics is look like this:')
    img,gt_bbox,gt_class,gt_score=data
    # print(type(img),type(gt_bbox),type(gt_class),type(gt_score))
    print(img.shape,gt_bbox.shape,gt_class.shape,gt_score.shape,'\n')
    # for item in data:
    #     if isinstance(item, torch.Tensor):
    #         print(item.shape)
    #     else:
    #         print(item)

