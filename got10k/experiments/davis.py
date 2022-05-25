import os
import json
import numpy as np

from got10k.datasets.davis import DAVIS
from got10k.experiments import ExperimentOTB
from got10k.utils.metrics import rect_iou


class ExperimentDAVISLike(ExperimentOTB):
    def __init__(self):
        self.use_confs = True

    def _record(self, record_file, boxes, times, confs):
        super()._record(record_file, boxes, times)
        # convert confs to string
        lines = ['%.4f' % c for c in confs]
        lines[0] = '99999.99'
        conf_file = record_file.replace(".txt", "_confidence.value")
        with open(conf_file, 'w') as f:
            f.write(str.join('\n', lines))

    def _calc_metrics(self, boxes, anno):
        ious = rect_iou(boxes, anno)
        # special case
        not_present_mask_anno = np.any(np.isnan(anno), axis=1)
        not_present_mask_boxes = np.any(np.isnan(boxes), axis=1)
        ious[np.logical_or(not_present_mask_anno, not_present_mask_boxes)] = 0.0
        ious[np.logical_and(not_present_mask_anno, not_present_mask_boxes)] = 1.0
        return ious

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)
            seq_ious = np.zeros((seq_num,))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                boxes = np.loadtxt(record_file, delimiter=',')
                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno do not match boxes' % seq_name)
                    assert len(boxes) == len(anno)

                conf_file = os.path.join(
                    self.result_dir, name, '%s_confidence.value' % seq_name)
                confs = np.loadtxt(conf_file)
                if not (len(boxes) == len(confs)):
                    print('warning: %s boxes do not match confs' % seq_name)
                    assert len(boxes) == len(confs)
                confs[0] = 99999.99

                # remove boxes for which confidence is -1?
                #boxes[confs == -1.0, :] = np.nan

                ious = self._calc_metrics(boxes, anno)
                miou = np.mean(ious[1:-1])
                seq_ious[s] = miou

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'miou': miou,
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            miou_total = np.mean(seq_ious)
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'miou': miou_total,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        return performance


class ExperimentDAVIS(ExperimentDAVISLike):
    def __init__(self, root_dir, result_dir='results', report_dir='reports', start_idx=0, end_idx=None,
                 version="2017_val"):
        self.dataset = DAVIS(root_dir, version)
        self.result_dir = os.path.join(result_dir, 'DAVIS' + str(version))
        self.report_dir = os.path.join(report_dir, 'DAVIS' + str(version))
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__()
