import os
import glob
import numpy as np
import PIL.Image
from .davis import DAVIS_Like, get_bbox_from_segmentation_mask_xywh


class YouTubeVOS(DAVIS_Like):
    def __init__(self, root_dir, version="valid", use_all_frames=True):
        img_version = version
        if use_all_frames:
            img_version += "_all_frames"
        img_dir = os.path.join(root_dir, img_version, "JPEGImages")
        ann_dir = os.path.join(root_dir, version, "Annotations")

        seqs = [x.split("/")[-2] for x in sorted(glob.glob(ann_dir + "/*/"))]

        seq_names = []
        self._start_times_and_ff_masks = []
        for seq in seqs:
            ids_to_start_time_and_ff_mask = {}
            start_annotation_files = sorted(glob.glob(os.path.join(ann_dir, seq, "*.png")))
            for start_annotation_file in start_annotation_files:
                t = int(start_annotation_file.split("/")[-1].replace(".png", ""))
                ann = np.array(PIL.Image.open(start_annotation_file))
                ann_ids = np.setdiff1d(np.unique(ann), [0])
                for id_ in ann_ids:
                    if id_ in ids_to_start_time_and_ff_mask:
                        if t < ids_to_start_time_and_ff_mask[id_][0]:
                            ids_to_start_time_and_ff_mask[id_] = (t, ann == id_)
                    else:
                        ids_to_start_time_and_ff_mask[id_] = (t, ann == id_)
            ids = sorted(ids_to_start_time_and_ff_mask.keys())
            for id_ in ids:
                seq_with_id = seq + "__" + str(id_)
                seq_names.append(seq_with_id)
                self._start_times_and_ff_masks.append(ids_to_start_time_and_ff_mask[id_])
        super().__init__(img_dir, ann_dir, seq_names)

    def __getitem__(self, index):
        r"""
        Args:
          index (integer or string): Index of a sequence.

        Returns:
          tuple: (img_files, anno), where ``img_files`` is a list of
              file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        start_time, ff_mask = self._start_times_and_ff_masks[index]
        seq_name = self.seq_names[index]
        sp = seq_name.split("__")
        img_files = sorted(glob.glob(os.path.join(self.img_dir, sp[0], "*.jpg")))
        img_files = [x for x in img_files if int(x.split("/")[-1].replace(".jpg", "")) >= start_time]
        anno = np.full((len(img_files), 4), fill_value=np.nan)
        anno[0] = get_bbox_from_segmentation_mask_xywh(ff_mask)
        return img_files, anno
