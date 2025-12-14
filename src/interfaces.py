from abc import ABC, abstractmethod
from typing import Any, List, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------
# Universal Pipeline Adapter
# -----------------------------
class PipelineAdapter(BaseEstimator, TransformerMixin):
    """
    General adapter for any pipeline-like object (transformer, model, pipeline)
    to make it sklearn-compatible with deep get_params support.
    """

    def __init__(self, obj):
        """
        obj: any object with fit/transform/predict interface
        """
        self.obj = obj

    def fit(self, X, y=None):
        if hasattr(self.obj, "fit_transform"):
            self.obj.fit_transform(X, y)
        else:
            self.obj.fit(X, y)
        return self

    def transform(self, X):
        if hasattr(self.obj, "transform"):
            return self.obj.transform(X)
        return X

    def predict(self, X):
        if hasattr(self.obj, "predict"):
            return self.obj.predict(X)
        raise AttributeError("Underlying object has no predict method")

    def predict_proba(self, X):
        if hasattr(self.obj, "predict_proba"):
            return self.obj.predict_proba(X)
        raise AttributeError("Underlying object has no predict_proba method")

    def get_params(self, deep=True):
        if hasattr(self.obj, "get_params"):
            if deep:
                return {f"obj__{k}": v for k, v in self.obj.get_params(deep=True).items()}
            else:
                return {"obj": self.obj}
        return {"obj": self.obj}


# -----------------------------
# Base class
# -----------------------------
class BasePreProcessor(ABC):
    """
    Base interface for all preprocessing steps.
    """

    def __init__(self):
        self._feature_names_out: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, Any] = None):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, Any] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self._feature_names_out

    @abstractmethod
    def get_description(self) -> str:
        pass

    def get_params(self, deep=True) -> dict:
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        if deep:
            for k, v in params.copy().items():
                if isinstance(v, BasePreProcessor):
                    nested = v.get_params(deep=True)
                    for nk, nv in nested.items():
                        params[f"{k}__{nk}"] = nv
        return params


# -----------------------------
# ValueModifier interface
# -----------------------------
class ValueModifierPreProcessor(BasePreProcessor):
    """
    Interface for transformers that modify values
    but do not change feature count or names.
    """
    pass


# -----------------------------
# ShapeChanger interface
# -----------------------------
class ShapeChangerPreProcessor(BasePreProcessor):
    """
    Interface for transformers that may change the number
    or names of features.
    """

    def __init__(self):
        super().__init__()
        self._input_features: List[str] = []
        self._new_features: List[str] = []
        self._removed_features: List[str] = []

    @abstractmethod
    def get_original_features(self) -> List[str]:
        return self._input_features

    @abstractmethod
    def get_new_features(self) -> List[str]:
        return self._new_features

    @abstractmethod
    def get_removed_features(self) -> List[str]:
        return self._removed_features

class BaseModel(ABC):
    """
    Base interface for any predictive model.
    Methods: fit, predict, optionally predict_proba.
    Compatible with universal PipelineAdapter for sklearn integration.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any):
        """
        Fit the model to data X and labels y.
        """
        return self

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict labels for X.
        """
        pass

    def predict_proba(self, X: Any) -> Any:
        """
        Optional: return predicted probabilities for X.
        """
        raise NotImplementedError("This model does not support probability predictions")

    @abstractmethod
    def get_params(self, deep=True) -> dict:
        """
        Return model parameters (hyperparameters).
        deep=True for nested objects.
        """
        return {}
