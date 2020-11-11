from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline

import torch

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.utils import get_tabular_preprocessers


class TabularColumnTransformer(autoPyTorchTabularPreprocessingComponent):
    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.random_state = random_state
        self.column_transformer: Optional[ColumnTransformer] = None

    def get_column_transformer(self) -> ColumnTransformer:
        """
        Get fitted column transformer that is wrapped around
        the sklearn early_preprocessor. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted sklearn column transformer
        """
        if self.column_transformer is None:
            raise AttributeError(
                "{} can't return column transformer before transform is called".format(self.__class__.__name__)
            )
        return self.column_transformer

    def fit(self, X: Dict[str, Any], y: Any = None) -> "TabularColumnTransformer":
        """
        Creates a column transformer for the chosen tabular
        preprocessors
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            "TabularColumnTransformer": an instance of self
        """
        self.check_requirements(X, y)

        numerical_pipeline = "drop"
        categorical_pipeline = "drop"

        preprocessors = get_tabular_preprocessers(X)
        if len(X["numerical_columns"]):
            numerical_pipeline = make_pipeline(*preprocessors["numerical"])
        if len(X["categorical_columns"]):
            categorical_pipeline = make_pipeline(*preprocessors["categorical"])

        self.column_transformer = make_column_transformer(
            (numerical_pipeline, X["numerical_columns"]),
            (categorical_pipeline, X["categorical_columns"]),
            remainder="passthrough",
        )
        self.column_transformer.fit(X["X_train"])

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the column transformer to fit dictionary
        Args:
            X (Dict[str, Any]): fit dictionary

        Returns:
            X (Dict[str, Any]): updated fit dictionary
        """
        X.update({"tabular_transformer": self.column_transformer})
        return X

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:

        if self.column_transformer is None:
            raise ValueError(
                "cant call {} without fitting the column transformer first.".format(self.__class__.__name__)
            )
        try:
            X = self.column_transformer.transform(X)
        except ValueError as msg:
            raise ValueError("{} in {}".format(msg, self.__class__))
        return X

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        super().check_requirements(X, y)
        if "numerical_columns" not in X or "categorical_columns" not in X:
            raise ValueError(
                "To fit a column transformer on tabular data"
                ", the fit dictionary must contain a list of "
                "the numerical and categorical columns of the "
                "data but only contains {}".format(X.keys())
            )
