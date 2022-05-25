# -*- coding: utf-8 -*-
# File: eval.py

import itertools
import random
import sys
import os
import json
import PIL
import numpy as np
import glob
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import cv2
import pycocotools.mask as cocomask
import tqdm
import tensorflow as tf
import xmltodict

from tensorpack.callbacks import Callback
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm

from common import CustomResize, clip_boxes, box_to_point8, point8_to_box
from data import get_eval_dataflow
from dataset import DetectionDataset
from config import config as cfg

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def predict_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.
    在一个图像上运行检测，使用TF可调用。
    这个函数应该在内部处理预处理。

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def predict_image_track_with_precomputed_ref_features(img, ref_features, model_func):
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img, ref_features)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def predict_image_track(img, ref_img, ref_bbox, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.
    在一个图像上运行检测，使用TF可调用。
    这个函数应该在内部处理预处理。

    Args:
        img: an image
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    resized_ref_img, params = resizer.augment_return_params(ref_img)

    ref_points = box_to_point8(ref_bbox[np.newaxis])
    ref_points = resizer.augment_coords(ref_points, params)
    resized_ref_boxes = point8_to_box(ref_points)
    resized_ref_bbox = resized_ref_boxes[0]

    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img, resized_ref_img, resized_ref_bbox)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def predict_dataflow(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id) 生成(image, image_id)的数据流
        model_func: a callable from the TF model. 一个TF模型的可调用对象
            It takes image and returns (boxes, probs, labels, [masks])
            它获取图像并返回(boxes, probs, labels， [mask])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.
            在多个评估实例之间共享的tqdm对象。如果为None，将创建一个新的。

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for ref_img, ref_bbox, target_img, target_bbox, gt_file in df:
            results = predict_image_track(target_img, ref_img, ref_bbox, model_func)
            all_results.append((gt_file, results, target_bbox))
            tqdm_bar.update(1)
    return all_results


def multithread_predict_dataflow(dataflows, model_funcs):
    """
    Running multiple `predict_dataflow` in multiple threads, and aggregate the results.
    在多个线程中运行多个' predict_dataflow '，并聚合结果

    Args:
        dataflows: a list of DataFlow to be used in :func:`predict_dataflow`
        model_funcs: a list of callable to be used in :func:`predict_dataflow`

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    num_worker = len(model_funcs)
    assert len(dataflows) == num_worker
    if num_worker == 1:
        return predict_dataflow(dataflows[0], model_funcs[0])
    kwargs = {'thread_name_prefix': 'EvalWorker'} if sys.version_info.minor >= 6 else {}
    with ThreadPoolExecutor(max_workers=num_worker, **kwargs) as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, model_funcs):
            futures.append(executor.submit(predict_dataflow, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results


class EvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.
    一次运行求值的回调。
    支持多gpu评估
    """

    _chief_only = False

    def __init__(self, eval_dataset, in_names, out_names, output_dir):
        self._eval_dataset = eval_dataset
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
            buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]

            # Use two predictor threads per GPU to get better throughput
            self.num_predictor = num_gpu if buggy_tf else num_gpu * 2
            self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
            self.dataflows = [get_eval_dataflow(self._eval_dataset,
                                                shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Only eval on the first machine.
            # Alternatively, can eval on all ranks and use allgather, but allgather sometimes hangs
            self._horovod_run_eval = hvd.rank() == hvd.local_rank()
            if self._horovod_run_eval:
                self.predictor = self._build_predictor(0)
                self.dataflow = get_eval_dataflow(self._eval_dataset,
                                                  shard=hvd.local_rank(), num_shards=hvd.local_size())

            self.barrier = hvd.allreduce(tf.random_normal(shape=[1]))

    def _build_predictor(self, idx):
        return self.trainer.get_predictor(self._in_names, self._out_names, device=idx)

    def _before_train(self):
        eval_period = cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        if cfg.TRAINER == 'replicated':
            all_results = multithread_predict_dataflow(self.dataflows, self.predictors)
        else:
            filenames = [os.path.join(
                logdir, 'outputs{}-part{}.json'.format(self.global_step, rank)
            ) for rank in range(hvd.local_size())]

            if self._horovod_run_eval:
                local_results = predict_dataflow(self.dataflow, self.predictor)
                fname = filenames[hvd.local_rank()]
                with open(fname, 'w') as f:
                    json.dump(local_results, f)
            self.barrier.eval()
            if hvd.rank() > 0:
                return
            all_results = []
            for fname in filenames:
                with open(fname, 'r') as f:
                    obj = json.load(f)
                all_results.extend(obj)
                os.unlink(fname)

        output_file = os.path.join(
            logdir, '{}-outputs{}.json'.format(self._eval_dataset, self.global_step))

        scores = DetectionDataset().eval_or_save_inference_results(
            all_results, self._eval_dataset, output_file)
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()
