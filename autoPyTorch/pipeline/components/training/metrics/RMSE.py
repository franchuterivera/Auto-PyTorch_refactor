from typing import Any, Dict, Optional

from pytorch_lightning.metrics import regression
from pytorch_lightning.metrics.metric import Metric

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class RMSE(autoPyTorchMetric):
    def __init__(self,
                 reduction: str = 'elementwise_mean',
                 ):
        super().__init__()
        self.reduction = reduction
        self.metric: Metric = regression.RMSE(reduction=self.reduction)

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'RMSE',
            'name': 'Root Mean Squared Error',
            'task_type': 'regression'
        }
