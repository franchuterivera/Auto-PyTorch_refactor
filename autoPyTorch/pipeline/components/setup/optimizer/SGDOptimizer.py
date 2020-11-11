from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np

from torch.optim import SGD

from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer import BaseOptimizerComponent


class SGDOptimizer(BaseOptimizerComponent):
    """
    Implements Stochstic Gradient Descend  algorithm.

    Args:
        lr (float): learning rate (default: 1e-2)
        momentum (float): momentum factor (default: 0)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self, lr: float, momentum: float, weight_decay: float, random_state: Optional[np.random.RandomState] = None,
    ):

        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseOptimizerComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        self.optimizer = SGD(
            params=X["network"].parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum,
        )

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            "shortname": "SGD",
            "name": "Stochastic gradient descent (optionally with momentum)",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The learning rate for the model
        lr = UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, default_value=1e-2, log=True)

        weight_decay = UniformFloatHyperparameter("weight_decay", lower=0.0, upper=0.1, default_value=0.0)

        momentum = UniformFloatHyperparameter("momentum", lower=0.0, upper=0.99, default_value=0.0)

        cs.add_hyperparameters([lr, weight_decay, momentum])

        return cs
