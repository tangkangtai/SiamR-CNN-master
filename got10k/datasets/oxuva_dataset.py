import os
import numpy as np
import PIL.Image


class Oxuva:
    def __init__(self, root_dir, subset="test"):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        task_file = os.path.join(root_dir, "tasks", subset + ".csv")
        with open(task_file) as f:
            import oxuva
            self._tasks = oxuva.load_dataset_tasks_csv(f)
        self.seq_names = [t[0] + "___" + t[1] for t in self._tasks]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer): Index of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        seq_name = self.seq_names[index]
        k = tuple(seq_name.split("___"))
        assert len(k) == 2, k
        task = self._tasks[k]
        r = task.init_rect
        x0 = r["xmin"]
        x1 = r["xmax"]
        y0 = r["ymin"]
        y1 = r["ymax"]
        w = x1 - x0
        h = y1 - y0
        anno_01 = np.array([x0, y0, w, h])
        imgs = []
        vid = k[0]
        for t in range(task.init_time, task.last_time + 1):
            img = os.path.join(self.root_dir, self.subset, "images", self.subset, vid, "%06d.jpeg" % t)
            imgs.append(img)
        w, h = PIL.Image.open(imgs[0]).size
        # scale anno with image size...
        anno = anno_01 * np.array([w, h, w, h], dtype=np.float32)
        return imgs, anno[np.newaxis]

    def __len__(self):
        return len(self.seq_names)


if __name__ == "__main__":
    # test code for dataset
    dataset = Oxuva("/globalwork/data/oxuva/")
    for imgs, annos in dataset:
        print(imgs, annos)
