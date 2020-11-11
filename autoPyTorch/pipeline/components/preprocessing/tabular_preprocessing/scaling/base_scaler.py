from typing import Any, Dict

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent,
)


class BaseScaler(autoPyTorchTabularPreprocessingComponent):
    """
    Provides abstract class interface for Scalers in AutoPytorch
    """

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted early_preprocessor into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        if self.preprocessor["numerical"] is None and self.preprocessor["categorical"] is None:
            raise ValueError("cant call transform on {} without fitting first.".format(self.__class__.__name__))
        X.update({"scaler": self.preprocessor})
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
        if "numerical_columns" not in X:
            raise ValueError(
                "To fit a scaler, the fit dictionary "
                "must contain a list of the numerical "
                "columns of the data but only contains {}".format(X.keys())
            )
