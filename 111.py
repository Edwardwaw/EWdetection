from voc import *
from coco import *
from operators import *
from batch_operators import *
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


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

gridop=GridMaskOp(use_h=True,use_w=True,rotate=1,offset=False,ratio=0.5,mode=1,prob=0.7,upper_iter=5000)

trans=[DecodeImage(to_rgb=True,with_mixup=False,with_cutmix=False,with_mosaic=True),
       MosaicImage(),
       # CutmixImage(alpha=1.5,beta=1.5),
       # MixupImage(alpha=1.5,beta=1.5),
       # AutoAugmentImage(autoaug_type='v1'),
       # RandomErasingImage(),
       # gridop,
       # ColorDistort(),
       # RandomExpand(fill_value=[123.675,116.28,103.53]),
       # RandomCrop(),
       # RandomFlipImage(is_normalized=False),
       NormalizeBox(),
       # PadBox(num_max_boxes=50),
       # BboxXYXY2XYWH(),
       # RandomShape(sizes=[320,352,384,416,448,512],random_inter=True),

       # NormalizeImage(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],is_scale=True,is_channel_first=False),
       # ResizeImage(target_size=800,max_size=1333,interp=1,use_cv2=True),
       # RandomScaledCrop(target_dim=512,scale_range=[0.1,2.0],interp=1),
       # Permute(to_bgr=False,channel_first=True),
       # PadBatch(32,False),
       # ToTensor(['gt_bbox', 'gt_class', 'gt_score'])
       ]



voc_dataset=VOCDataSet('VOC2007','JPEGImages','Annotations',split='test',with_background=False,
                   class_aware_sampling=False,cutmix=0.0,mosaic=0.5,
                   inputs_def={'fields':['image', 'im_info', 'im_id','gt_bbox', 'gt_class', 'gt_score','is_crowd']},
                   sample_transforms=trans,remove_empty=True)

# coco_dataset=COCODataSet('coco2017','val2017','annotations/instances_val2017.json',with_background=False,
#                     class_aware_sampling=False,mosaic=0.6,
#                     inputs_def={'fields': ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'gt_score', 'is_crowd']},
#                     sample_transforms=trans,remove_empty=True
#                     )




i=0
for sample in iter(voc_dataset):
    i+=1
    print('the ',i,'st image')
    # print(len(sample))
    image=sample[0]
    image=cv2.cvtColor(image.astype('uint8'),cv2.COLOR_RGB2BGR)
    h,w=sample[0].shape[:2]
    # print(sample[3].shape)
    if sample[3].shape[0]==0:
        break
    for box in sample[3]:
        if np.sum(box)==0:
            continue
        else:
            x1=int((box[0]-box[2]/2.)*w)
            x2=int((box[0]+box[2]/2.)*w)
            y1=int((box[1]-box[3]/2.)*h)
            y2 = int((box[1] + box[3] / 2.)*h)
            x1,y1,x2,y2=box*np.array([w,h,w,h])
            if x1<=-1 or x2<= -1 or y1<= -1 or y2<= -1:
                print(x1, y1, x2, y2)
                break
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255))

            # x1,y1,x2,y2=box
            # print(x1, y1, x2, y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    # cv2.imshow('tet',image)
    # cv2.waitKey(30)
    cv2.imwrite('./save/%s.jpg'%i,image)
    # gridop.set_iter(i)
    # break

# print(len(coco_dataset))



# dataloader=DataLoader(dataset,batch_size=4,num_workers=4,collate_fn=collate_fn)
#
# for i,data in enumerate(dataloader):
#     print('the ',i,'st batch input data, its statistics is look like this:\n')
#     for item in data:
#         if isinstance(item, torch.Tensor):
#             print(item.shape)
#         else:
#             print(item)

