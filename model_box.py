# -*- coding: utf-8 -*-
# File: model_box.py

import numpy as np
from collections import namedtuple
import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope

from config import config


@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, x y x y
        window: [h, w]
    """
    # tf.maximum(x,y, name=None)返回 张量x 和 张量y 的最大值 组成的一个张量(即表达式：x > y ? x : y)
    # 该函数将返回一个张量并且与 x 具有相同的类型.
    boxes = tf.maximum(boxes, 0.0)
    # tf.tile(input,multiples,name=None)
    # 通过平铺(tile)给定的张量来构造张量,

    # tf.reverse(tensor,axis,name=None) 反转张量的特定维度
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,) 或者 (4,1)
    # 返回 x 和 y 的最小值(即表达式：x < y ? x : y).
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes

# ？？？？？？？？？？？？？？？？？？
# https://www.jianshu.com/p/9ac545f0e3e8
# 在预测过程中，网络输出的坐标形式是（dx,dy,dw,dh），
# 而真实值的坐标形式是（y1,x1,y2,x2），
# 同样要把网络输出和 anchors 结合，decode成真实值的形式
@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))

    # tf.split(value, num_or_size_splits,axis=0,num=None,name='split')将张量分割成子张量
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)

    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    # 计算 (anchor/bbox)?  的宽，高和中心坐标
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    # _C.PREPROC.MAX_SIZE = 1333
    # 边框的长和宽的最高值
    clip = np.log(config.PREPROC.MAX_SIZE / 16.)

    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya

    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
    # tf.concat(values, axis, name='concat')
    # 将张量沿一个维度串联
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


# 在训练过程中，groundtruth 的坐标形式是（y1,x1,y2,x2），
# 而网络输出的坐标形式是（dx,dy,dw,dh），
# 两者之间的表示形式不同，因此要将 groundtruth 和 anchors 结合，
# 来表示anchor_layers 上每个位置的分类 label 和坐标偏移 loc（即网络的输出），即 encode 过程
# 作者：huim


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32
        anchors: (..., 4), float32

    Returns:
        box_encoded: (..., 4), float32 with the same shape.
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)

    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    # 注意，这里并不是所有框都有效。有些可能是零
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes

    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))


@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    tf.image.crop_and_resize 的对齐版本,跟随我们定义的floating point boxes

    # tf.image.crop_and_resize(image,boxes,box_ind,crop_size,...)
    # 从输入图像张量中提取crop(裁剪),并双线调整它们的大小(可能高宽比变化)到由crop_size指定的通用输出大小
    # image: 表示特征图
    # boxes：指需要划分的区域，输入格式为[ymin，xmin，ymax，xmax]
    # 设crop的区域坐标是[y1,x1,y2,x2]，那么想要得到相应正确的crop图形就一定要归一化,即图片的长度是[w,h],则实际输入的boxes为[y1/h,x1/w,y2/h,x2/w]。
    # 不归一化的话，会自动进行补齐，超出1的就全变成成黑色框了。
    # box_ind: 是boxes和image之间的索引
    # crop_size: 表示RoiAlign之后的大小。
    # 版权声明：本文为CSDN博主「Kun Li」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/u012193416/article/details/86525411
    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    # tf.stop_gradient(input,name=None) 停止梯度计算
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        这个函数将 fpcoor box 转换为 tf.image.crop_and_resize 所使用的格式
        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    # tf.image.crop_and_resize(image,boxes,box_ind,crop_size,...)
    # 从输入图像张量中提取crop(裁剪),并双线调整它们的大小(可能高宽比变化)到由crop_size指定的通用输出大小
    # image: 表示特征图
    # boxes：指需要划分的区域，输入格式为[ymin，xmin，ymax，xmax]
    # 设crop的区域坐标是[y1,x1,y2,x2]，那么想要得到相应正确的crop图形就一定要归一化,即图片的长度是[w,h],则实际输入的boxes为[y1/h,x1/w,y2/h,x2/w]。
    # 不归一化的话，会自动进行补齐，超出1的就全变成成黑色框了。
    # box_ind: 是boxes和image之间的索引
    # crop_size: 表示RoiAlign之后的大小。
    # 版权声明：本文为CSDN博主「Kun Li」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/u012193416/article/details/86525411
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret

# roi_align
@under_name_scope()
def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution

    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    tfshape = tf.shape(boxes)[0]

    ret = crop_and_resize(
        featuremap, boxes,
        # tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        tf.zeros(shape=tfshape, dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret


class RPNAnchors(namedtuple('_RPNAnchors', ['boxes', 'gt_labels', 'gt_boxes'])):
    """
    boxes (FS x FS x NA x 4): The anchor boxes.
    gt_labels (FS x FS x NA):
    gt_boxes (FS x FS x NA x 4): Groundtruth boxes corresponding to each anchor. 
    """                         # groundtruth box 对应的每个 anchor
    def encoded_gt_boxes(self):
        return encode_bbox_target(self.gt_boxes, self.boxes)

    def decode_logits(self, logits):
        return decode_bbox_target(logits, self.boxes)

    @under_name_scope()
    def narrow_to(self, featuremap):
        """
        Slice anchors to the spatial size of this featuremap.
        """
        # [x,y,w,h]
        shape2d = tf.shape(featuremap)[2:]  # h,w

        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)
        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        gt_labels = tf.slice(self.gt_labels, [0, 0, 0], slice3d)
        gt_boxes = tf.slice(self.gt_boxes, [0, 0, 0, 0], slice4d)
        return RPNAnchors(boxes, gt_labels, gt_boxes)


if __name__ == '__main__':
    """
    Demonstrate what's wrong with tf.image.crop_and_resize:
    """
    import tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()

    # want to crop 2x2 out of a 5x5 image, and resize to 4x4
    image = np.arange(25).astype('float32').reshape(5, 5)
    boxes = np.asarray([[1, 1, 3, 3]], dtype='float32')
    target = 4

    print(crop_and_resize(
        image[None, None, :, :], boxes, [0], target)[0][0])
    """
    Expected values:
    4.5 5 5.5 6
    7 7.5 8 8.5
    9.5 10 10.5 11
    12 12.5 13 13.5

    You cannot easily get the above results with tf.image.crop_and_resize.
    Try out yourself here:
    """
    print(tf.image.crop_and_resize(
        image[None, :, :, None],
        np.asarray([[1, 1, 2, 2]]) / 4.0, [0], [target, target])[0][:, :, 0])
