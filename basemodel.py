# -*- coding: utf-8 -*-
# File: basemodel.py

import numpy as np
from contextlib import ExitStack, contextmanager
import tensorflow as tf

from tensorpack.models import BatchNorm, Conv2D, MaxPooling, layer_register
from tensorpack.tfutils import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.varreplace import custom_getter_scope, freeze_variables

from config import config as cfg

# GroupNorm 组归一化
# GroupNorm将channel分组，然后再做归一化；
@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):

    # 四维张量(N,C,H,W)
    shape = x.get_shape().as_list()
    ndims = len(shape)

    assert ndims == 4, shape

    # 通道数
    chan = shape[1]
    assert chan % group == 0, chan

    # 组归一化的  大小
    group_size = chan // group

    orig_shape = tf.shape(x)

    h, w = orig_shape[2], orig_shape[3]

    # tf.reshape(tensor , shape)  ## shape也是一个tensor
    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))


    # tf.nn: 提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
    # tf.layers：主要提供的高层的神经网络，主要和卷积相关的，tf.nn会更底层一些。
    # tf.contrib：tf.contrib.layers提供够将计算图中的 网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变
    # ----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 通过在axes上聚合x的内容来计算均值和方差
    # tf.nn.moments(x, axes, shift=None, name=None,keep_dims = False)
    # x：一个Tensor.
    # axes：整数数组.用于计算均值和方差的轴.
    # shift：未在当前实现中使用
    # name：用于计算moment的操作范围的名称.
    # keep_dims：产生与输入具有相同维度的moment.

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    #  tf.constant_initializer：常量初始化函数
    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())

    # tf.reshape(tensor, shape)
    # 给定tensor,这个操作返回一个张量,它与带有形状shape的tensor具有相同的值.
    # tensor：一个Tensor.
    # shape：一个Tensor；必须是以下类型之一：int32,int64；用于定义输出张量的形状.
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    # tf.nn.batch_normalization(x,mean,variance,offset,scale,variance_epsilon,name=None)
    # x：任意维度的输入Tensor.
    # mean：一个平均Tensor.
    # variance：一个方差Tensor.
    # offset：一个偏移量Tensor,通常在方程式中表示为\(\ beta \),或者为None；如果存在,将被添加到归一化张量.
    # scale：一个标度Tensor,通常在方程式中表示为\(\ gamma \),或为None；如果存在,则将比例应用于归一化张量.
    # variance_epsilon：一个小的浮点数,以避免除以0.
    # name：此操作的名称(可选).
    # 返回：标准化,缩放,偏移张量

    # 输出公式 y=scale∗(x−mean)/var+offset
    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


# 单星号函数参数。单星号函数参数接收的参数组成一个元组。
# 双星号函数参数。双星号函数参数接收的参数组成一个字典
def freeze_affine_getter(getter, *args, **kwargs):
    # custom getter to freeze affine params inside bn
    # 自定义getter来冻结bn中的仿射参数
    name = args[0] if len(args) else kwargs.get('name')
    if name.endswith('/gamma') or name.endswith('/beta'):
        kwargs['trainable'] = False
        ret = getter(*args, **kwargs)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, ret)
    else:
        ret = getter(*args, **kwargs)
    return ret


def maybe_reverse_pad(topleft, bottomright):
    if cfg.BACKBONE.TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]


@contextmanager
def backbone_scope(freeze):
    """
    Args:
        freeze (bool): whether to freeze all the variables under the scope
    """
    # nonlin 非线性
    def nonlin(x):
        #
        x = get_norm()(x)
        # 计算校正线性：max(features, 0)
        return tf.nn.relu(x)

    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'), \
         argscope(Conv2D, use_bias=False, activation=nonlin,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
            ExitStack() as stack:
        # contextlib模块的ExitStack()动态管理退出回调堆栈的上下文管理器。
        if cfg.BACKBONE.NORM in ['FreezeBN', 'SyncBN']:
            if freeze or cfg.BACKBONE.NORM == 'FreezeBN':
                stack.enter_context(argscope(BatchNorm, training=False))
            else:
                stack.enter_context(argscope(
                    BatchNorm, sync_statistics='nccl' if cfg.TRAINER == 'replicated' else 'horovod'))

        if freeze:
            stack.enter_context(freeze_variables(stop_gradient=False, skip_collection=True))
        else:
            # the layers are not completely freezed, but we may want to only freeze the affine
            if cfg.BACKBONE.FREEZE_AFFINE:
                stack.enter_context(custom_getter_scope(freeze_affine_getter))
        yield

# 图像预处理
def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            # 将张量类型转换为新类型
            image = tf.cast(image, tf.float32)
            #pixel 像素 # 像素均值？
        mean = cfg.PREPROC.PIXEL_MEAN
        # 将输入转换为数组。        # pixel_std 像素标准差
        std = np.asarray(cfg.PREPROC.PIXEL_STD)
        if bgr:
            mean = mean[::-1]   # 逆序
            std = std[::-1]     # 逆序
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd
        return image


def get_norm(zero_init=False):
    if cfg.BACKBONE.NORM == 'None':
        return lambda x: x
    if cfg.BACKBONE.NORM == 'GN':
        Norm = GroupNorm
        layer_name = 'gn'
    else:
        Norm = BatchNorm
        layer_name = 'bn'
    return lambda x: Norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)

# tf.identity返回一个与输入具有相同形状和内容的张量
# resnet的shortcut
def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.shape[1]
    if n_in != n_out:   # change dimension when channel is not the same
        # TF's SAME mode output ceil(x/stride), which is NOT what we want when x is odd and stride is 2
        # In FPN mode, the images are pre-padded already.
        if not cfg.MODE_FPN and stride == 2:
            l = l[:, :, :-1, :-1]
        return Conv2D('convshortcut', l, n_out, 1,strides=stride, activation=activation)
    else:
        return l

# resnet的bottleneck
# bottleneck先通过一个1x1的卷积减少通道数，使得中间卷积的通道数减少为1/4；
# 中间的普通卷积做完卷积后输出通道数等于输入通道数；第三个卷积用于增加（恢复）通道数，
# 使得bottleneck的输出通道数等于bottleneck的输入通道数。
# 这两个1x1卷积有效地较少了卷积的参数个数和计算量。
#
def resnet_bottleneck(l, ch_out, stride):

    shortcut = l
    if cfg.BACKBONE.STRIDE_1X1:
        if stride == 2:
            l = l[:, :, :-1, :-1]
        # 手动笔记
        # Conv2D( variable_scope_name
        #         inputs,
        #  4D张量的格式为[batch, height, width, channels]，分别为图片的批量（每次处理的图片张数），
        #  图片高度像素数，图片宽度的像素数，图片通道数（彩色为3通道，灰度为一通道，其余可能还有深度等）
        #         filters,
        #  [filter_height, filter_width, in_channels, out_channels]，
        #  分别为卷积核/滤波器的像素高度，像素宽度，输入通道数（与input中的通道数相等），输出通道数（卷积核个数，卷积层学习的特征个数）
        #         kernel_size,
        #         strides=(1, 1), 四维张量，[----,高度步长，宽度步长，----],第一和第四个数没用，但是默认为1
        #         padding='same',
        #         )
        l = Conv2D('conv1', l, ch_out, 1, strides=stride)
        l = Conv2D('conv2', l, ch_out, 3, strides=1)
    else:
        l = Conv2D('conv1', l, ch_out, 1, strides=1)

        if stride == 2:
            # tf.pad(tensor,paddings,mode='CONSTANT',name=None)
            # padings ，代表每一维填充多少行/列，它的维度一定要和tensor的维度是一样的，这里的维度不是传统上数学维度，
            # 如[[2,3,4],[4,5,6]]是一个3乘4的矩阵，但它依然是二维的，所以pad只能是[[1,2],[1,2]]这种。
            #
            # mode 可以取三个值，分别是"CONSTANT" ,“REFLECT”,“SYMMETRIC”
            # mode=“CONSTANT” 填充0
            # mode="REFLECT"映射填充，上下（1维）填充顺序和paddings是相反的，左右（零维）顺序补齐
            # mode="SYMMETRIC"对称填充，上下（1维）填充顺序是和paddings相同的，左右（零维）对称补齐
            # ————————————————
            # 版权声明：本文为CSDN博主「路上的病人」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https://blog.csdn.net/qq_40994943/article/details/85331327
            l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
            l = Conv2D('conv2', l, ch_out, 3, strides=2, padding='VALID')
        else:
            l = Conv2D('conv2', l, ch_out, 3, strides=stride)

    if cfg.BACKBONE.NORM != 'None':

        l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_norm(zero_init=True))
    else:
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=tf.identity,
                   kernel_initializer=tf.constant_initializer())

    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_norm(zero_init=False))
    return tf.nn.relu(ret, name='output')


def resnet_group(name, l, block_func, features, count, stride):

    #tensorflow.variable_scope(...)
    # 用于定义创建变量(层)的操作的上下文管理器.
    # 此上下文管理器验证(可选)values是否来自同一图形,确保图形是默认的图形,并推送名称范围和变量范围.
    # 如果name_or_scope不是None,则使用as is.如果scope是None,则使用default_name.在这种情况下,
    # 如果以前在同一范围内使用过相同的名称,则通过添加_N来使其具有唯一性.
    # 变量范围允许您创建新变量并共享已创建的变量,同时提供检查以防止意外创建或共享.
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_c4_backbone(image, num_blocks):

    assert len(num_blocks) == 3

    freeze_at = cfg.BACKBONE.FREEZE_AT

    with backbone_scope(freeze=freeze_at > 0):

        l = tf.pad(image, [[0, 0], [0, 0], maybe_reverse_pad(2, 3), maybe_reverse_pad(2, 3)])
        l = Conv2D('conv0', l, 64, 7, strides=2, padding='VALID')

        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
        # Same as `tf.layers.MaxPooling2D`. Default strides is equal to pool_size.
        l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')

    with backbone_scope(freeze=freeze_at > 1):
        c2 = resnet_group('group0', l, resnet_bottleneck, 64, num_blocks[0], 1)

    with backbone_scope(freeze=False):
        c3 = resnet_group('group1', c2, resnet_bottleneck, 128, num_blocks[1], 2)
        c4 = resnet_group('group2', c3, resnet_bottleneck, 256, num_blocks[2], 2)

    # 16x downsampling up to now
    # downsampling 缩减像素采样
    return c4


@auto_reuse_variable_scope
def resnet_conv5(image, num_block):

    with backbone_scope(freeze=False):
        l = resnet_group('group3', image, resnet_bottleneck, 512, num_block, 2)
        return l


def resnet_fpn_backbone(image, num_blocks):
    # _C.BACKBONE.FREEZE_AT = 4  # options: 0, 1, 2
    freeze_at = cfg.BACKBONE.FREEZE_AT
    shape2d = tf.shape(image)[2:]
    # _C.FPN.RESOLUTION_REQUIREMENT = _C.FPN.ANCHOR_STRIDES[3]  # [3] because we build FPN with features r2,r3,r4,r5
    # _C.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
    mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)
    # tf.ceil()返回不小于 x 的元素最小整数.
    new_shape2d = tf.cast(tf.ceil(tf.cast(shape2d, tf.float32) / mult) * mult, tf.int32)
    pad_shape2d = new_shape2d - shape2d
    assert len(num_blocks) == 4, num_blocks

    with backbone_scope(freeze=freeze_at > 0):
        chan = image.shape[1]
        pad_base = maybe_reverse_pad(2, 3)
        l = tf.pad(image, tf.stack(
            [[0, 0], [0, 0],
             [pad_base[0], pad_base[1] + pad_shape2d[0]],
             [pad_base[0], pad_base[1] + pad_shape2d[1]]]))
        l.set_shape([None, chan, None, None])
        l = Conv2D('conv0', l, 64, 7, strides=2, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
        # MaxPooling Same as `tf.layers.MaxPooling2D`. Default strides is equal to pool_size
        # MaxPooling(variable_scope_name, inputs, pool_size, strides=None, padding='valid', data_format='channels_last')
        # tf.layers.max_polling2d(inputs,pool_size,strides,padding='valid',data_format='channels_last',name=None)
        # inputs: 进行池化的数据
        # pool_size:池化核的大小(pool_height,pool_width),如[3,3],也可以设为pool_size= 3
        # strides: 池化的滑动步长，可以设置为[1,1]这样的两个整数，也可以直接设置为一个整数 strides=2
        # padding:边缘填充，'same'和'valid'选其一，默认为valid
        # dta_format: 输入数据格式，默认为channels_last ，即 (batch, height, width, channels),
        # 也可以设置为channels_first 对应 (batch, channels, height, width)
        # name: 层的名字
        l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')

    with backbone_scope(freeze=freeze_at > 1):
        c2 = resnet_group('group0', l, resnet_bottleneck, 64, num_blocks[0], 1)

    with backbone_scope(freeze=freeze_at > 2):
        c3 = resnet_group('group1', c2, resnet_bottleneck, 128, num_blocks[1], 2)
        c4 = resnet_group('group2', c3, resnet_bottleneck, 256, num_blocks[2], 2)
        c5 = resnet_group('group3', c4, resnet_bottleneck, 512, num_blocks[3], 2)
    # 32x downsampling up to now
    # size of c5: ceil(input/32)
    return c2, c3, c4, c5
