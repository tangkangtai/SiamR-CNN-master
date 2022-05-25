import os
import glob
import numpy as np
import PIL.Image
from functools import partial
from multiprocessing import Pool


class DAVIS_Like:
    def __init__(self, img_dir, ann_dir, seq_names, multiobject=True):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.seq_names = seq_names
        self._multiobject = multiobject

    def __getitem__(self, index):
        r"""
        Args:
          index (integer or string): Index of a sequence.

        Returns:
          tuple: (img_files, anno), where ``img_files`` is a list of
              file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        seq_name_maybe_with_obj_id = self.seq_names[index]
        sp = seq_name_maybe_with_obj_id.split("__")
        seq_name_raw = sp[0]
        if len(sp) > 1 and self._multiobject:
            obj_id = int(sp[1])
        else:
            obj_id = None
        img_files = sorted(glob.glob(os.path.join(self.img_dir, seq_name_raw, "*.jpg")))
        ann_files = [x.replace(self.img_dir, self.ann_dir).replace(".jpg", ".png") for x in img_files]

        with Pool(8) as pool:
            anno = pool.map(partial(png_to_rect, obj_id=obj_id), ann_files)
        anno = np.stack(anno)
        assert len(img_files) == len(anno), (len(img_files), len(anno))
        assert anno.shape[1] == 4, anno.shape[1]
        return img_files, anno

    def __len__(self):
        return len(self.seq_names)


class DAVIS(DAVIS_Like):
    def __init__(self, root_dir, version="2017_val"):
        seq_names = []
        seq_names_filename = os.path.join(root_dir, "ImageSets", "2017", version + "_ids.txt")
        with open(seq_names_filename) as f:
            for l in f:
                seq_names.append(l.strip())
        img_dir = os.path.join(root_dir, "JPEGImages", "480p")
        ann_dir = os.path.join(root_dir, "Annotations", "480p")
        super().__init__(img_dir, ann_dir, seq_names, multiobject=version != "2016_val")


def png_to_rect(ann_filename, obj_id):
    if not os.path.exists(ann_filename):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    ann = np.array(PIL.Image.open(ann_filename))
    if obj_id is None:
        ann = (ann > 0).astype(np.uint8)
    else:
        ann = (ann == obj_id).astype(np.uint8)
    if ann.any():
        return get_bbox_from_segmentation_mask_xywh(ann)
    else:
        return np.array([np.nan, np.nan, np.nan, np.nan])


def get_bbox_from_segmentation_mask_xywh(mask):
    object_locations = (np.stack(np.where(np.equal(mask, 1))).T[:, :2]).astype(np.int32)
    y0 = np.min(object_locations[:, 0])
    x0 = np.min(object_locations[:, 1])
    y1 = np.max(object_locations[:, 0]) + 1
    x1 = np.max(object_locations[:, 1]) + 1
    w = x1 - x0
    h = y1 - y0
    bbox = np.stack([x0, y0, w, h])
    return bbox
