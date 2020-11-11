from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.utils import check_random_state

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class autoPyTorchChoice(object):
    """Allows for the dynamically generation of components as pipeline steps.

    Args:
        dataset_properties (Dict[str, Union[str, int]]): Describes the dataset
            to work on
        random_state (Optional[np.random.RandomState]): allows to produce reproducible
            results by setting a seed for randomized settings

    Attributes:
        random_state (Optional[np.random.RandomState]): allows to produce reproducible
            results by setting a seed for randomized settings
        choice (autoPyTorchComponent): the choice of components for this stage
    """

    def __init__(self, dataset_properties: Dict[str, Any], random_state: Optional[np.random.RandomState] = None):

        # Since all calls to get_hyperparameter_search_space will be done by the
        # pipeline on construction, it is not necessary to construct a
        # configuration space at this location!
        # self.configuration = self.get_hyperparameter_search_space(
        #     dataset_properties).get_default_configuration()

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        self.dataset_properties = dataset_properties
        self._check_dataset_properties(dataset_properties)
        # Since the pipeline will initialize the hyperparameters, it is not
        # necessary to do this upon the construction of this object
        # self.set_hyperparameters(self.configuration)
        self.choice = None

    def get_components(cls: "autoPyTorchChoice") -> Dict[str, autoPyTorchComponent]:
        """Returns and ordered dict with the components available
        for current step.

        Args:
            cls (autoPyTorchChoice): The choice object from which to query the valid
                components

        Returns:
            Dict[str, autoPyTorchComponent]: The available components via a mapping
                from the module name to the component class

        """
        raise NotImplementedError()

    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, autoPyTorchComponent]:
        """
        Wrapper over get components to incorporate include/exclude
        user specification

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Results:
            Dict[str, autoPyTorchComponent]: A dictionary with valid components for this
                choice object

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: " "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(
        self, configuration: Configuration, init_params: Optional[Dict[str, Any]] = None
    ) -> "autoPyTorchChoice":
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration): which configuration to apply to
                the chosen component
            init_params (Optional[Dict[str, any]]): Optional arguments to
                initialize the chosen component

        Returns:
            self: returns an instance of self
        """
        new_params = {}

        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

        for param, value in params.items():
            param = param.replace(choice, "").replace(":", "")
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        new_params["random_state"] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[Dict[str, str]] = None,
        default: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            default: (Optional[str]) : Default component to use in hyperparameters
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                chosen component
        """
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> autoPyTorchComponent:
        """Handy method to check if a component is fitted

        Args:
            X (np.ndarray): the input features
            y (np.ndarray): the target features
        """
        # Allows to use check_is_fitted on the choice object
        self.fitted_ = True
        if kwargs is None:
            kwargs = {}
        assert self.choice is not None, "Cannot call fit without initializing the component"
        return self.choice.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target given an input, by using the chosen component

        Args:
            X (np.ndarray): input features from which to predict the target

        Returns:
            np.ndarray: the predicted target
        """
        assert self.choice is not None, "Cannot call predict without initializing the component"
        return self.choice.predict(X)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the current choice in the fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            (Dict[str, Any])
        """
        assert self.choice is not None, "Can not call transform without initialising the component"
        return self.choice.transform(X)

    def _check_dataset_properties(self, dataset_properties: Dict[str, Any]) -> None:
        """
        A mechanism in code to ensure the correctness of the initialised dataset properties.
        Args:
            dataset_properties:

        """
        assert isinstance(dataset_properties, dict), "dataset_properties must be a dictionary"
