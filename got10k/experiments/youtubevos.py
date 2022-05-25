import os

from got10k.datasets.youtubevos import YouTubeVOS
from got10k.experiments.davis import ExperimentDAVISLike


class ExperimentYouTubeVOS(ExperimentDAVISLike):
    def __init__(self, root_dir, result_dir='results', report_dir='reports', start_idx=0, end_idx=None,
                 version="val"):
        self.dataset = YouTubeVOS(root_dir, version)
        self.result_dir = os.path.join(result_dir, 'YouTubeVOS_' + str(version))
        self.report_dir = os.path.join(report_dir, 'YouTubeVOS_' + str(version))
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__()
