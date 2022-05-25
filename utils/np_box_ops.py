# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Operations for [N, 4] numpy arrays representing bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape, dtype='f4'),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape, dtype='f4'),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between box collections.
      计算box集合之间的成对 交叉和并
  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding M boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
    形状为[N, M]的numpy数组表示成对iou得分
  """
  intersect = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  # expand_dims(a, axis)中，a为numpy数组，axis为需添加维度的轴
  # 就是在axis的那一个轴上把数据加上去
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect

  return intersect / union


def ioa(boxes1, boxes2):
  """Computes pairwise intersection-over-area between box collections.
   计算box集合之间的两两交叉的面积。

  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).
    两个盒子之间的交叉过面积(ioa)定义为盒子1和盒子2的交叉面积。
    注意，ioa不是对称的，即ioa (box1, box2) != ioa (box2, box1)。

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
    形状为[N, M]的numpy数组，表示成对ioa分数。
  """
  intersect = intersection(boxes1, boxes2)
  inv_areas = np.expand_dims(1.0 / area(boxes2), axis=0)
  return intersect * inv_areas
