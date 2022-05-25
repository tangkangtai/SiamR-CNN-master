import glob
import os
import numpy as np


class CustomDataset(object):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        seqs = sorted(glob.glob(self.root_dir + "/*"))
        seqs = [x for x in seqs if os.path.isdir(x)]
        self.seq_names = seqs
        assert len(seqs) > 0
        print("custom dataset seqs", seqs)

    def __getitem__(self, index):
        assert 0 <= index <= len(self.seq_names)
        seq = self.seq_names[index]
        imgs = sorted(glob.glob(os.path.join(self.root_dir, seq, "*.jpg")))
        gt_file = os.path.join(self.root_dir, seq, "groundtruth.txt")
        with open(gt_file) as f:
            l = f.readline().strip()
            sp = l.split()
            assert len(sp) == 4
            x, y, w, h = [float(x) for x in sp]
            bbox = np.array([x, y, w, h])
        return imgs, bbox[np.newaxis]

    def __len__(self):
        return len(self.seq_names)
