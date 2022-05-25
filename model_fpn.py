# -*- coding: utf-8 -*-

import itertools
import numpy as np
import tensorflow as tf

from tensorpack.models import Conv2D, FixedUnPooling, MaxPooling, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.tower import get_current_tower_context

from basemodel import GroupNorm
from config import config as cfg
from model_box import roi_align
from model_rpn import generate_rpn_proposals, rpn_losses
from utils.box_ops import area as tf_area

# FPN（feature pyramid networks）特征金字塔网络目标检测
# 作者一方面将FPN放在RPN网络中用于生成proposal，原来的RPN网络是以主网络的某个卷积层输出的feature map作为输入，
# 简单讲就是只用这一个尺度的feature map。但是现在要将FPN嵌在RPN网络中，生成  不同尺度特征并融合  作为RPN网络的输入。
# 在每一个scale层，都定义了不同大小的anchor，对于P2，P3，P4，P5，P6这些层，
# 定义anchor的大小为32^2,64^2,128^2,256^2，512^2，另外每个scale层都有3个长宽对比度：1:2，1:1，2:1。
# 所以整个特征金字塔有15种anchor
# https://blog.csdn.net/baidu_30594023/article/details/82623623
@layer_register(log_shape=True)
def fpn_model(features):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features
    # _C.FPN.NUM_CHANNEL = 256
    num_channel = cfg.FPN.NUM_CHANNEL
    # GroupNorm
    use_gn = cfg.FPN.NORM == 'GN'

    def upsample2x(name, x):
        # FixedUnPooling(x, shape, unpool_mat=None, data_format='channels_last')
        # 用一个固定矩阵对输入进行解池，以执行kronecker(克罗内克内积 )乘积。

        return FixedUnPooling(
            name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
            data_format='channels_first')

        # tf.image.resize is, again, not aligned.
        # with tf.name_scope(name):
        #     shape2d = tf.shape(x)[2:]
        #     x = tf.transpose(x, [0, 2, 3, 1])
        #     x = tf.image.resize_nearest_neighbor(x, shape2d * 2, align_corners=True)
        #     x = tf.transpose(x, [0, 3, 1, 2])
        #     return x

    with argscope(Conv2D, data_format='channels_first',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=1.)):
        # # 手动笔记
        #         # Conv2D( variable_scope_name,
        #         #         inputs,
        #         #  4D张量的格式为[batch, height, width, channels]，分别为图片的批量（每次处理的图片张数），
        #         #  图片高度像素数，图片宽度的像素数，图片通道数（彩色为3通道，灰度为一通道，其余可能还有深度等）
        #         #         filters,
        #         #  [filter_height, filter_width, in_channels, out_channels]，
        #         #  分别为卷积核/滤波器的像素高度，像素宽度，输入通道数（与input中的通道数相等），输出通道数（卷积核个数，卷积层学习的特征个数）
        #         #         kernel_size,
        #         #         strides=(1, 1), 四维张量，[----,高度步长，宽度步长，----],第一和第四个数没用，但是默认为1
        #         #         padding='same',
        #         #         )

        # 经过卷积之后的 特征图
        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1) for i, c in enumerate(features)]

        if use_gn:
            # groupNorm 组归一化处理
            lat_2345 = [GroupNorm('gn_c{}'.format(i + 2), c) for i, c in enumerate(lat_2345)]

        lat_sum_5432 = []

        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])

                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3)
                 for i, c in enumerate(lat_sum_5432[::-1])]
        if use_gn:
            p2345 = [GroupNorm('gn_p{}'.format(i + 2), c) for i, c in enumerate(p2345)]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first', padding='VALID')

        return p2345 + [p6]


@under_name_scope()
def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes (nx4):

    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.cast(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),   # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


@under_name_scope()
def multilevel_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            boxes_on_featuremap = boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois


@under_name_scope()
def neck_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    aligned_features = None
    for i in range(4):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            boxes_on_featuremap = rcnn_boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            level_features = roi_align(features[i], boxes_on_featuremap, resolution)
            if aligned_features is None:
                aligned_features = level_features
            else:
                aligned_features += level_features
    return aligned_features


def multilevel_rpn_losses(
        multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Args:
        multilevel_anchors: #lvl RPNAnchors
        multilevel_label_logits: #lvl tensors of shape HxWxA
        multilevel_box_logits: #lvl tensors of shape HxWxAx4

    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl], multilevel_box_logits[lvl],
                name_scope='level{}'.format(lvl + 2))
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2], name='label_loss')
        total_box_loss = tf.add_n(losses[1::2], name='box_loss')
        add_moving_summary(total_label_loss, total_box_loss)
    return [total_label_loss, total_box_loss]


@under_name_scope()
def generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d):
    """
    Args:
        multilevel_pred_boxes: #lvl HxWxAx4 boxes
        multilevel_label_logits: #lvl tensors of shape HxWxA

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_pred_boxes) == num_lvl
    assert len(multilevel_label_logits) == num_lvl

    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]
                proposal_boxes, proposal_scores = generate_rpn_proposals(
                    tf.reshape(pred_boxes_decoded, [-1, 4]),
                    tf.reshape(multilevel_label_logits[lvl], [-1]),
                    image_shape2d, fpn_nms_topk)
                all_boxes.append(proposal_boxes)
                all_scores.append(proposal_scores)

        proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4
        proposal_scores = tf.concat(all_scores, axis=0)  # n
        # Here we are different from Detectron.
        # Detectron picks top-k within the batch, rather than within an image. However we do not have a batch.
        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)
    else:
        for lvl in range(num_lvl):
            with tf.name_scope('Lvl{}'.format(lvl + 2)):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]
                all_boxes.append(tf.reshape(pred_boxes_decoded, [-1, 4]))
                all_scores.append(tf.reshape(multilevel_label_logits[lvl], [-1]))
        all_boxes = tf.concat(all_boxes, axis=0)
        all_scores = tf.concat(all_scores, axis=0)
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            all_boxes, all_scores, image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if training else cfg.RPN.TEST_POST_NMS_TOPK)

    tf.sigmoid(proposal_scores, name='probs')  # for visualization
    return tf.stop_gradient(proposal_boxes, name='boxes'), \
        tf.stop_gradient(proposal_scores, name='scores')