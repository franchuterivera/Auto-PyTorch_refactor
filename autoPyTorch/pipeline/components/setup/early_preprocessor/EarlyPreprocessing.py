from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.pipeline.components.setup.early_preprocessor.utils import get_preprocess_transforms, preprocess


class EarlyPreprocessing(autoPyTorchSetupComponent):
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> "EarlyPreprocessing":
        self.check_requirements(X, y)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        transforms = get_preprocess_transforms(X)

        if X["is_small_preprocess"]:
            X["X_train"] = preprocess(dataset=X["X_train"], transforms=transforms)
        else:
            X.update({"preprocess_transforms": transforms})
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
        if "is_small_preprocess" not in X:
            raise ValueError(
                "To preprocess data, the fit dictionary "
                "must contain whether the data is small "
                "enough to preprocess as is_small_preprocess "
                "but only contains {}".format(X.keys())
            )

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        return ConfigurationSpace()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            "shortname": "Preprocessing",
            "name": "Preprocessing Node",
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop("random_state", None)
        string += " (" + str(info) + ")"
        return string
