# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from six.moves import range

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

# faster rcnn生成anchor
# base_size:
#           这个参数指定了最初的类似感受野的区域大小，因为经过多层卷积池化之后，
#           feature map上一点的感受野对应到原始图像就会是一个区域，这里设置的是16，也就是feature map上一点对应到原图的大小为16x16的区域。也可以根据需要自己设置

# ratios:
#           这个参数指的是要将16x16的区域，按照1:2,1:1,2:1三种比例进行变换

# scales:
#             这个参数是要将输入的区域，的宽和高进行三种倍数，2^3=8，2^4=16，2^5=32倍的放大，
#             如16x16的区域变成(16*8)*(16*8)=128*128的区域，(16*16)*(16*16)=256*256的区域，(16*32)*(16*32)=512*512的区域，


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # 表示最基本的一个大小为16x16的区域，四个值，分别代表这个区域的左上角和右下角的点的坐标。
    base_anchor = np.array([1, 1, base_size, base_size], dtype='float32') - 1
    """base_anchor得值为[0, 0, 15, 15]"""

    # 这一句是将前面的16x16的区域进行ratio变化，也就是输出三种宽高比的anchors
    ratio_anchors = _ratio_enum(base_anchor, ratios)

    # 进行完上面的宽高比变换之后，接下来执行的是面积的scale变换，

    # 这里最重要的是_scale_enum函数，该函数定义如下，对上一步得到的ratio_anchors中的三种宽高比的anchor，
    # 再分别进行三种scale的变换，也就是三种宽高比，搭配三种scale，最终会得到9种宽高比和scale 的anchors。这就是论文中每一个点对应的9种anchor
    # ————————————————
    # scale_enum函数中也是首先将宽高比变换后的每一个ratio_anchor转化成（宽，高，中心点横坐标，中心点纵坐标）的形式，
    # 再对宽和高均进行scale倍的放大，然后再转换成四个坐标值的形式。最终经过宽高比和scale变换得到的9种尺寸的anchors的坐标如下

    # numpy.vstack() 参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组
    # 它是垂直（按照行顺序）的把数组给堆叠起来
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

# 其主要作用是将输入的anchor的四个坐标值转化成（宽，高，中心点横坐标，中心点纵坐标）的形式。
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

# 输入参数为一个anchor(四个坐标值表示)和三种宽高比例（0.5,1,2）
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)

    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
