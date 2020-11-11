from typing import Any, Callable, Dict, List, Optional

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
from autoPyTorch.pipeline.components.setup.network.utils import (
    shake_drop,
    shake_drop_get_bl,
    shake_get_alpha_beta,
    shake_shake,
)


class ResNet(BaseNetworkComponent):
    """
    Implementation of a Residual Network builder

    Args:
        num_groups (int): total number of groups in the network. A group is composed of
                          blocks_per_group ResNets.
        intermediate_activation (str): type of activation for this layer
        random_state (Optional[np.random.RandomState]): random state
        num_units_%d (int): Number of units of layer %d
        blocks_per_group_%d (int): Number of ResNet group %d will have
        use_dropout (bool): Whether or not to add dropout at each layer
        use_shake_shake (bool): Whether or not to use use_shake_shake regularization
        use_shake_drop (bool): Whether to use shake drop regularization
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
        layers.append(torch.nn.Linear(in_features, self.config["num_units_0"]))

        # build num_groups-1 groups each consisting of blocks_per_group ResBlocks
        # the output features of each group is defined by num_units_i
        for i in range(1, self.config["num_groups"] + 1):
            layers.append(
                self._add_group(
                    in_features=self.config["num_units_%d" % (i - 1)],
                    out_features=self.config["num_units_%d" % i],
                    blocks_per_group=self.config["blocks_per_group_%d" % i],
                    last_block_index=(i - 1) * self.config["blocks_per_group_%d" % i],
                    dropout=self.config["use_dropout"],
                )
            )

        layers.append(torch.nn.BatchNorm1d(self.config["num_units_%i" % self.config["num_groups"]]))
        layers.append(ResNet.get_activations_dict()[self.intermediate_activation]())

        layers.append(torch.nn.Linear(self.config["num_units_%i" % self.config["num_groups"]], out_features))

        network = torch.nn.Sequential(*layers)
        return network

    def _add_group(
        self, in_features: int, out_features: int, blocks_per_group: int, last_block_index: int, dropout: bool
    ) -> torch.nn.Module:
        """
        Adds a group into the main network.
        In the case of ResNet a group is a set of blocks_per_group
        Resblocks

        Args:
            in_features (int): number of inputs for the current block
            out_features (int): output dimensionality for the current block
            blocks_per_group (int): Number of ResNet per group
            last_block_index (int): block index for shake regularization
            droupout (bool): whether or not use dropout
        """
        blocks = list()
        for i in range(blocks_per_group):
            blocks.append(
                ResBlock(
                    config=self.config,
                    in_features=in_features,
                    out_features=out_features,
                    blocks_per_group=blocks_per_group,
                    block_index=last_block_index + i,
                    dropout=dropout,
                    activation=ResNet.get_activations_dict()[self.intermediate_activation],
                )
            )
            in_features = out_features
        return torch.nn.Sequential(*blocks)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            "shortname": "ResNet",
            "name": "Residual Network",
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        min_num_gropus: int = 1,
        max_num_groups: int = 9,
        min_blocks_per_groups: int = 1,
        max_blocks_per_groups: int = 4,
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

        intermediate_activation = CategoricalHyperparameter(
            "intermediate_activation", choices=list(ResNet.get_activations_dict().keys())
        )
        cs.add_hyperparameters([num_groups, intermediate_activation])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter("use_dropout", choices=[True, False])
        cs.add_hyperparameters([use_dropout])

        use_shake_shake = CategoricalHyperparameter("use_shake_shake", choices=[True, False])
        use_shake_drop = CategoricalHyperparameter("use_shake_drop", choices=[True, False])
        shake_drop_prob = UniformFloatHyperparameter("max_shake_drop_probability", lower=0.0, upper=1.0)
        cs.add_hyperparameters([use_shake_shake, use_shake_drop, shake_drop_prob])
        cs.add_condition(CS.EqualsCondition(shake_drop_prob, use_shake_drop, True))

        # It is the upper bound of the nr of groups,
        # since the configuration will actually be sampled.
        for i in range(0, max_num_groups + 1):

            n_units = UniformIntegerHyperparameter("num_units_%d" % i, lower=min_num_units, upper=max_num_units,)
            blocks_per_group = UniformIntegerHyperparameter(
                "blocks_per_group_%d" % i, lower=min_blocks_per_groups, upper=max_blocks_per_groups
            )

            cs.add_hyperparameters([n_units, blocks_per_group])

            if i > 1:
                cs.add_condition(CS.GreaterThanCondition(n_units, num_groups, i - 1))
                cs.add_condition(CS.GreaterThanCondition(blocks_per_group, num_groups, i - 1))

            this_dropout = UniformFloatHyperparameter("dropout_%d" % i, lower=0.0, upper=1.0)
            cs.add_hyperparameters([this_dropout])

            dropout_condition_1 = CS.EqualsCondition(this_dropout, use_dropout, True)

            if i > 1:

                dropout_condition_2 = CS.GreaterThanCondition(this_dropout, num_groups, i - 1)

                cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
            else:
                cs.add_condition(dropout_condition_1)
        return cs


class ResBlock(torch.nn.Module):
    """
    __author__ = "Max Dippel, Michael Burkart and Matthias Urban"
    """

    def __init__(
        self,
        config: Dict[str, Any],
        in_features: int,
        out_features: int,
        blocks_per_group: int,
        block_index: int,
        dropout: bool,
        activation: torch.nn.Module,
    ):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation

        self.shortcut = None
        self.start_norm = None  # type: Optional[Callable]

        # if in != out the shortcut needs a linear layer to match the result dimensions
        # if the shortcut needs a layer we apply batchnorm and activation to the shortcut
        # as well (start_norm)
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
            self.start_norm = torch.nn.Sequential(torch.nn.BatchNorm1d(in_features), self.activation())

        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config["num_groups"]
        self.layers = self._build_block(in_features, out_features)

        if config["use_shake_shake"]:
            self.shake_shake_layers = self._build_block(in_features, out_features)

    # each bloack consists of two linear layers with batch norm and activation
    def _build_block(self, in_features: int, out_features: int) -> torch.nn.Module:
        layers = list()

        if self.start_norm is None:
            layers.append(torch.nn.BatchNorm1d(in_features))
            layers.append(self.activation())
        layers.append(torch.nn.Linear(in_features, out_features))

        layers.append(torch.nn.BatchNorm1d(out_features))
        layers.append(self.activation())

        if self.config["use_dropout"]:
            layers.append(torch.nn.Dropout(self.dropout))
        layers.append(torch.nn.Linear(out_features, out_features))

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x

        # if shortcut is not none we need a layer such that x matches the output dimension
        if self.shortcut is not None and self.start_norm is not None:
            # in this case self.start_norm is also != none
            # apply start_norm to x in order to have batchnorm+activation
            # in front of shortcut and layers. Note that in this case layers
            # does not start with batchnorm+activation but with the first linear layer
            # (see _build_block). As a result if in_features == out_features
            # -> result = x + W(~D(A(BN(W(A(BN(x))))))
            # if in_features != out_features
            # -> result = W_shortcut(A(BN(x))) + W_2(~D(A(BN(W_1(A(BN(x))))))
            x = self.start_norm(x)
            residual = self.shortcut(x)

        if self.config["use_shake_shake"]:
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            x = shake_shake(x1, x2, alpha, beta)
        else:
            x = self.layers(x)

        if self.config["use_shake_drop"]:
            alpha, beta = shake_get_alpha_beta(self.training, x.is_cuda)
            bl = shake_drop_get_bl(
                self.block_index,
                1 - self.config["max_shake_drop_probability"],
                self.num_blocks,
                self.training,
                x.is_cuda,
            )
            x = shake_drop(x, alpha, beta, bl)

        x = x + residual
        return x
