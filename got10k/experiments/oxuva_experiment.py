from .otb import ExperimentOTB
from ..datasets import oxuva_dataset
#import oxuva

import os


class ExperimentOxuva(ExperimentOTB):
    def __init__(self, root_dir, subset='test', result_dir='results', report_dir='reports', start_idx=0, end_idx=None):
        self.dataset = oxuva_dataset.Oxuva(root_dir, subset)
        self.result_dir = os.path.join(result_dir, 'Oxuva' + str(subset))
        self.report_dir = os.path.join(report_dir, 'Oxuva' + str(subset))
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.start_idx = start_idx
        self.end_idx = end_idx
        #if end_idx is None:
        #    self.end_idx = 167
        self.use_confs = True
        #self.dump_as_csv = True
        # new approach: dump as default and then postproc with script to get it to right format
        self.dump_as_csv = False
        self.has_groundtruth = False

    #def _record(self, record_file, boxes, times, confs):
        #TODO
    #    oxuva.dump_predictions_csv()
