from typing import Any, Dict

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.base_image_preprocessor import (
    autoPyTorchImagePreprocessingComponent,
)


class BaseNormalizer(autoPyTorchImagePreprocessingComponent):
    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({"normalise": self})
        return X

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        super().check_requirements(X, y)
        if "channelwise_mean" not in X or type(X["channelwise_mean"]) != np.ndarray:
            raise ValueError(
                "To normalise, the fit dictionary "
                "must contain channelwise_mean of type "
                "np.ndarray but only contains {}".format(X.keys())
            )
        if "channelwise_std" not in X or type(X["channelwise_std"]) != np.ndarray:
            raise ValueError(
                "To normalise, the fit dictionary "
                "must contain channelwise_std of type "
                "np.ndarray but only contains {}".format(X.keys())
            )
        if 0 in X["channelwise_std"]:
            raise ZeroDivisionError("Can't normalise when std is zero")

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop("random_state", None)
        string += " (" + str(info) + ")"
        return string
