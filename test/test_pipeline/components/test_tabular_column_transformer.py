import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.compose import ColumnTransformer

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import (
    TabularColumnTransformer,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import (
    EncoderChoice,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class TabularPipeline(TabularClassificationPipeline):
    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]],) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties = {"target_type": "tabular_classification"}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend(
            [
                ("imputer", SimpleImputer()),
                ("encoder", EncoderChoice(default_dataset_properties)),
                ("scaler", ScalerChoice(default_dataset_properties)),
                ("tabular_transformer", TabularColumnTransformer()),
            ]
        )
        return steps


class TabularTransformerTest(unittest.TestCase):
    def test_tabular_preprocess_only_numerical(self):
        X = dict(
            X_train=np.random.random((10, 15)),
            is_small_preprocess=True,
            numerical_columns=list(range(15)),
            categorical_columns=[],
            categories=[],
            num_features=15,
            num_classes=2,
        )
        dataset_properties = dict(numerical_columns=list(range(15)), categorical_columns=[],)
        pipeline = TabularPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        column_transformer = X["tabular_transformer"]

        # check if transformer was added to fit dictionary
        self.assertIn("tabular_transformer", X.keys())
        # check if transformer is of expected type
        self.assertIsInstance(column_transformer, ColumnTransformer)

        data = column_transformer.fit_transform(X["X_train"])
        self.assertIsInstance(data, np.ndarray)

    def test_tabular_preprocess_only_categorical(self):
        X = dict(
            X_train=np.array([["male", "germany"], ["female", "germany"], ["male", "germany"]], dtype=object),
            is_small_preprocess=True,
            numerical_columns=[],
            categorical_columns=list(range(2)),
            categories=[["male", "female"], ["germany"]],
            num_features=15,
            num_classes=2,
        )
        dataset_properties = dict(numerical_columns=[], categorical_columns=list(range(2)))
        pipeline = TabularPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        column_transformer = X["tabular_transformer"]

        # check if transformer was added to fit dictionary
        self.assertIn("tabular_transformer", X.keys())
        # check if transformer is of expected type
        self.assertIsInstance(column_transformer, ColumnTransformer)

        data = column_transformer.fit_transform(X["X_train"])
        self.assertIsInstance(data, np.ndarray)


if __name__ == "__main__":
    unittest.main()
