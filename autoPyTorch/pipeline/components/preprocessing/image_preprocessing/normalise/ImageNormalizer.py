from typing import Any, Dict, Optional, Union

import numpy as np

import torch.tensor

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.base_normalizer import BaseNormalizer


class ImageNormalizer(BaseNormalizer):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        self.random_state = random_state
        self.mean = None  # type: Optional[np.ndarray]
        self.std = None  # type: Optional[np.ndarray]

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "ImageNormalizer":
        """
        Initialises early_preprocessor and returns self.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            autoPyTorchImagePreprocessingComponent: self
        """
        self.check_requirements(X, y)
        self.mean = X["channelwise_mean"]
        self.std = X["channelwise_std"]
        return self

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying early_preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.tensor]: Transformed data tensor
        """
        X = (X - self.mean) / self.std
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "normalize",
            "name": "Image Normalizer Node",
        }
