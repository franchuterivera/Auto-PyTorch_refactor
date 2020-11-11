from typing import Any, Dict, List, Optional

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import torch

from autoPyTorch.pipeline.components.setup.network.base_network import BaseNetworkComponent
from autoPyTorch.pipeline.components.setup.network.utils import get_shaped_neuron_counts


class ShapedMLPNet(BaseNetworkComponent):
    """
    Implementation of a Shapped MLP -- an MLP with the number of units
    arranged so that a given shape is honored

    Args:
        num_groups (int): total number of layers this MLP will have
        intermediate_activation (str): type of activation for this layer
        final_activation (str): the final activation of this class
        random_state (Optional[np.random.RandomState]): random state
        use_dropout (bool): Whether or not to add dropout at each layer
        dropout_%d (float): The assigned dropout of layer %d
        resnet_shape (str): A geometrical shape, that guides the construction
                            of the number of units per group.
    """

    def __init__(
        self,
        intermediate_activation: str,
        final_activation: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs: Any
    ):

        super().__init__(
            intermediate_activation=intermediate_activation,
            final_activation=final_activation,
            random_state=random_state,
        )
        self.config = kwargs

    def build_network(self, in_features: int, out_features: int) -> torch.nn.Module:
        """Returns the actual PyTorch model, that is dynamically created
        from a self.config object.

        self.config is a dictionary created form a given config in the config space.
        It contains the necessary information to build a network.
        """
        layers = list()  # type: List[torch.nn.Module]
        neuron_counts = get_shaped_neuron_counts(
            self.config["mlp_shape"], in_features, out_features, self.config["max_units"], self.config["num_groups"]
        )
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            dropout_shape = get_shaped_neuron_counts(self.config["mlp_shape"], 0, 0, 1000, self.config["num_groups"])

        previous = in_features
        for i in range(self.config["num_groups"] - 1):
            if i >= len(neuron_counts):
                break
            if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
                dropout = dropout_shape[i] / 1000 * self.config["max_dropout"]
            else:
                dropout = 0.0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]

        layers.append(torch.nn.Linear(previous, out_features))
        return torch.nn.Sequential(*layers)

    def _add_layer(self, layers: List[torch.nn.Module], in_features: int, out_features: int, dropout: float) -> None:
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(ShapedMLPNet.get_activations_dict()[self.intermediate_activation]())
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            layers.append(torch.nn.Dropout(dropout))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            "shortname": "ShapedMLP",
            "name": "Shaped Multi Layer Perceptron",
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        min_num_gropus: int = 1,
        max_num_groups: int = 15,
        min_num_units: int = 10,
        max_num_units: int = 1024,
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=min_num_gropus, upper=max_num_groups, default_value=5
        )

        mlp_shape = CategoricalHyperparameter(
            "mlp_shape", choices=["funnel", "long_funnel", "diamond", "hexagon", "brick", "triangle", "stairs"]
        )

        intermediate_activation = CategoricalHyperparameter(
            "intermediate_activation", choices=list(ShapedMLPNet.get_activations_dict().keys())
        )

        max_units = UniformIntegerHyperparameter("max_units", lower=min_num_units, upper=max_num_units,)

        cs.add_hyperparameters([num_groups, intermediate_activation, mlp_shape, max_units])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter("use_dropout", choices=[True, False])
        max_dropout = UniformFloatHyperparameter("max_dropout", lower=0.0, upper=1.0)
        cs.add_hyperparameters([use_dropout, max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
