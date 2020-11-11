import numbers
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

import torch

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseNetworkComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for networks
    in Auto-Pytorch"""

    def __init__(
        self,
        intermediate_activation: str,
        final_activation: Optional[str],
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self.network = None
        self.intermediate_activation = intermediate_activation
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
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

        in_features = X["num_features"]
        out_features = X["num_classes"]

        self.network = self.build_network(in_features, out_features)

        # Properly set the network training device
        self.to(self.device)

        return self

    @abstractmethod
    def build_network(self, in_feature: int, out_features: int) -> torch.nn.Module:
        """This method returns a pytorch network, that is dynamically built
        using:

            common network arguments from the base class:
                * intermediate_activation
                * final_activation

            a self.config that is network specific, and contains the additional
            configuration hyperparameters to build a domain specific network
        """
        raise NotImplementedError()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({"network": self.network})
        return X

    def get_network(self) -> torch.nn.Module:
        """Return the underlying network object.
        Returns:
            model : the underlying network object
        """
        assert self.network is not None, "No network was initialized"
        return self.network

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """ This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

        # For the Network, we need the number of input features,
        # to build the first network layer
        if "num_features" not in X.keys():
            raise ValueError(
                "Could not parse the number of input features in the fit dictionary "
                "To fit a network, the number of features is needed to define "
                "the hidden layers, yet the dict contains only: {}".format(X.keys())
            )

        assert isinstance(X["num_features"], numbers.Integral), "num_features: {}".format(type(X["num_features"]))

        # For the Network, we need the number of classes,
        # to build the last layer
        if "num_classes" not in X:
            raise ValueError(
                "Could not parse the number of classes in the fit dictionary "
                "To fit a network, the number of classes is needed to define "
                "the hidden layers, yet the dict contains only: {}".format(X.keys())
            )
        assert isinstance(X["num_classes"], numbers.Integral), "num_classes: {}".format(type(X["num_classes"]))

    @classmethod
    def get_activations_dict(cls) -> Dict[str, torch.nn.Module]:
        """
        This method highlights the activations that can be used,
        when dynamically building a network.
        """
        return {
            "relu": torch.nn.ReLU,
            "sigmoid": torch.nn.Sigmoid,
            "tanh": torch.nn.Tanh,
            "leakyrelu": torch.nn.LeakyReLU,
            "selu": torch.nn.SELU,
            "rrelu": torch.nn.RReLU,
            "tanhshrink": torch.nn.Tanhshrink,
            "hardtanh": torch.nn.Hardtanh,
            "elu": torch.nn.ELU,
            "prelu": torch.nn.PReLU,
        }

    def get_network_weights(self) -> torch.nn.parameter.Parameter:
        """Returns the weights of the network"""
        assert self.network is not None, "No network was initialized"
        return self.network.parameters()

    def to(self, device: Optional[torch.device] = None) -> None:
        """Setups the network in cpu or gpu"""
        assert self.network is not None, "No network was initialized"
        if device is not None:
            self.network = self.network.to(device)
        else:
            self.network = self.network.to(self.device)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.network.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop("network", None)
        info.pop("random_state", None)
        string += " (" + str(info) + ")"
        return string
