from __future__ import absolute_import

import os

from ..datasets.custom_dataset import CustomDataset

from .otb import ExperimentOTB


class ExperimentCustom(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for custom dataset.
    
    Args:
        root_dir (string): Root directory of TColor128 dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir,
                 result_dir='results', report_dir='reports', name='custom', start_idx=0, end_idx=None):
        self.dataset = CustomDataset(root_dir)
        self.result_dir = os.path.join(result_dir, name)
        self.report_dir = os.path.join(report_dir, name)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.use_confs = False
        self.dump_as_csv = False
        self.has_groundtruth = False
