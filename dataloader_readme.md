# 数据处理模块

说明：参考paddledetection的pytorch数据预处理模块库。

待完成事项：

- [ ] cutmix数据增强
- [ ] mAP 评估代码
- [ ] testloader编写

## 1.数据准备

目前支持`COCO`、`VOC`数据集的预处理，数据准备分为三步：数据解析、数据增强、组建batch数据，基本流程如下图片所示类似。

<img src="https://paddledetection.readthedocs.io/_images/reader_figure.png" alt="../_images/reader_figure.png" style="zoom: 50%;" />

### 数据解析

数据解析的功能实现在 `COCODataSet`和`VOCDataset`中，其解析的主要功能函数如下： 

| 方法                         | 输入 | 输出                    | 备注                                              |
| ---------------------------- | ---- | ----------------------- | ------------------------------------------------- |
| `load_roidb_and_cname2cid()` | 无   | 无                      | 加载数据集中Roidb数据源list, 类别名到id的映射dict |
| `get_roidb()`                | 无   | list[dict], Roidb数据源 | 获取数据源                                        |
| `get_cname2cid()`            | 无   | dict，类别名到id的映射  | 获取标签ID                                        |
| `get_anno()`                 | 无   | str, 标注文件路径       | 获取标注文件路径                                  |
| `get_imid2path()`            | 无   | dict, 图片路径          | 获取图片路径                                      |

说明：

`roidbs（a list of dict）`:整个数据集的annotation信息解析后保存在列表roidbs。其每一个元素是一个dict，保存一张图片的标注信息，其保存内容如下：

```python
xxx_rec = {
    'im_file': im_fname,         # 一张图像的完整路径
    'im_id': np.array([img_id]), # 一张图像的ID序号
    'h': im_h,                   # 图像高度
    'w': im_w,                   # 图像宽度
    'is_crowd': is_crowd,        # 是否是群落对象, 默认为0 (VOC中无此字段，为了使个别模型同时适配于COCO与VOC)
    'gt_class': gt_class,        # 标注框标签名称的ID序号
    'gt_bbox': gt_bbox,          # 标注框坐标(xmin, ymin, xmax, ymax)
    'gt_score': gt_score,        # 标注框置信度得分 (此字段为了适配Mixup操作)
    'gt_poly': gt_poly,          # 分割掩码，此字段只在coco_rec中出现，默认为None(暂不支持)
    'difficult': difficult       # 是否是困难样本，此字段只在voc_rec中出现，默认为0
}
```

`cname2cid (dict)` : 保存了类别名到id的映射的一个dict。在COCO数据集中，会根据[COCO API](https://github.com/cocodataset/cocoapi)自动加载`cname2cid`。在VOC数据集中，如果设置`use_default_label=False`，将从`label_list.txt`中读取类别列表， 反之则可以没有`label_list.txt`文件。`label_list.txt`的格式如下所示，每一行文本表示一个类别：

```python
aeroplane
bicycle
bird
...
```

`COCO`数据集的组织形式如下,根据标注文件路径（anno_path），调用[COCO API](https://github.com/cocodataset/cocoapi)加载并解析COCO格式数据源`roidbs`和`cname2cid`：

```python
coco/
├── annotations
│   ├── instances_train2014.json
│   ├── instances_train2017.json
│   ├── instances_val2014.json
│   ├── instances_val2017.json
│   │   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
│   │   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   │   ...
```

`VOC`数据集的组织形式如下:

```python
├──VOC
│   ├── Annotations
│       ├── 001789.xml
│       │   ...
│   ├── JPEGImages
│       ├── 001789.jpg
│       │   ...
│   ├── ImageSets
│       |   ...
```

新添加数据源

新建`XXXDataset.py`，定义类`XXXDataSet`，并重写`load_roidb_and_cname2cid`方法对`roidbs`与`cname2cid`更新：

```python
class XXXDataSet(DataSet):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 ...
                 ):
        self.roidbs = None
        self.cname2cid = None
        ...

    def load_roidb_and_cname2cid(self):
        ...
        省略具体解析数据逻辑
        ...
        self.roidbs, self.cname2cid = records, cname2cid
```

### 数据增强

本文件暂时支持如下数据增强算子：

| 名称                             | 作用                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| `DecodeImage`                    | 从图像文件或内存buffer中加载图像，格式为BGR、HWC格式，如果设置to_rgb=True，则转换为RGB格式。 |
| `ResizeImage`                    | 根据特定的插值方式调整图像大小                               |
| `RandomFlipImage`                | 随机水平翻转图像                                             |
| `NormalizeImage`                 | 对图像像素值进行归一化，如果设置is_scale=True，则先将像素值除以255.0，像素值缩放到到[0-1]区间。 |
| `RandomDistort`                  | 随机扰动图片亮度、对比度、饱和度和色相                       |
| `ExpandImage`                    | 将原始图片放入用像素均值填充(随后会在减均值操作中减掉)的扩张图中，对此图进行裁剪、缩放和翻转 |
| `CropImage`                      | 根据缩放比例、长宽比例生成若干候选框，再依据这些候选框和标注框的面积交并比(IoU)挑选出符合要求的裁剪结果 |
| `CropImageWithDataAchorSampling` | 基于CropImage，在人脸检测中，随机将图片尺度变换到一定范围的尺度，大大增强人脸的尺度变化 |
| `NormalizeBox`                   | 对bounding box进行归一化                                     |
| `Permute`                        | 对图像的通道进行排列并转为BGR格式。假如输入是HWC顺序，通道C上是RGB格式，设置channel_first=True，将变成CHW，设置to_bgr=True，通道C上变成BGR格式。 |
| `MixupImage`                     | 按比例叠加两张图像                                           |
| `RandomInterpImage`              | 使用随机的插值方式调整图像大小                               |
| `Resize`                         | 根据特定的插值方式同时调整图像与bounding box的大小           |
| `MultiscaleTestResize`           | 将图像重新缩放为多尺度list的每个尺寸                         |
| `ColorDistort`                   | 根据特定的亮度、对比度、饱和度和色相为图像增加噪声           |
| `NormalizePermute`               | 归一化图像并改变图像通道顺序                                 |
| `RandomExpand`                   | 原理同ExpandImage，以随机比例与角度对图像进行裁剪、缩放和翻转 |
| `RandomCrop`                     | 原理同CropImage，以随机比例与IoU阈值进行处理                 |
| `PadBox`                         | 如果bounding box的数量少于num_max_boxes，则将零填充到bbox    |
| `BboxXYXY2XYWH`                  | 将bounding box从(xmin,ymin,xmax,ymin)形式转换为(xmin,ymin,width,height)格式 |
| `CutmixImage`                    | 采用cutmix数据增强（暂未支持，欢迎提交request）              |
| `MosaicImage`                    | 采用mosaic数据增强                                           |
| `RandomShape`                    | 随机对图片进行resize操作                                     |
| `PadMultiScaleTest`              | 在多尺度测试中对图像进行填充                                 |
| `ToTensor`                       | 根据fields参数，将sample中相应的item由np.array转为torch.Tensor |

说明：上表中的数据增强算子的输入与输出都是单张图片`sample`，`sample`是由`{'image':xx, 'im_info': xxx, ...}`组成，来自于上文提到的`roidbs`中的字典信息。

### 组建batch数据

本库的`Dataset`类继承自`toch.utils.data.Dataset`，其实现的功能有如下3个：

| 功能                                               | 实现函数              |
| -------------------------------------------------- | --------------------- |
| 根据数据集中各目标类别的数量进行cls_aware_sampling | `cal_image_weight`    |
| 根据配置的数据增强参数sample_trans进行序列化的增强 | `Compose`             |
| 根据input_def参数，返回sample（dict）所需要的item  | `collect_needed_item` |

本库的`dataloader`采用`toch.utils.data.DataLoader`类，可根据关键字列表`fields`实现对sample中相应项的collate，并重新书写`sampler`函数原生支持多卡训练。

| 函数                 | 参数          | 功能                                                         |
| -------------------- | ------------- | ------------------------------------------------------------ |
| `collate_fn`         | batch, fields | 把a list of tuple的数据根据关键字列表fields组织为模型训练需要的形式,比如把n个size=[c,h,w]的tensor stack为size=[n,c,h,w]的tensor |
| `DistributedSampler` |               | 在分布式训练中，实现每个节点加载一个专有数据子集的方式。     |

## 2.配置及运行

首先实例化`DataSet`类

```python
coco_dataset=COCODataSet(dataset_dir='coco2017',         # 数据集根目录
                         image_dir='val2017',       # 图像数据基于数据集根目录的相对路径
                         anno_path='annotations/instances_val2017.json',   # 标注文件基于数据集根目录的相对路径
                         with_background=False,       # 背景是否作为一类标签
                         class_aware_sampling=False,      # 是否根据数据集中各目标类别的数据量来进行采样
                         mixup=0.0,      # 采用mixup augmentation的概率
                         cutmix=0.0,     #  采用cutmix augmentation的概率
                         mosaic=0.5,     #  采用mosaic augmentation的概率
                         remove_empty=True,      # 当遇到gt_box为空的样本时重新采样，直到gt_box不为空
                         sample_transforms=trans,     # 数据增强列表
                         inputs_def={'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']})      # 训练中所需要的/需要返回的sample (dict)中的item
```

其中，`trans`是采用的数据增强算子的实例化列表,配置方法可参考`paddledetection` 的配置文件

```python
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
```

最后把参数传入`make_dataloader`函数，得到实例化之后的`dataloader`

```python
dataloader=make_dataloader(dataset, batch_size=8, num_workers=8, is_train=True, distributed=False,
                   											  max_iter=None, start_iter=0)
```



