import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate

from fraud_detection.config import get_settings

logger = logging.getLogger(__name__)

class FraudModelTrainer:
    """Trains machine learning models for fraud detection."""

    MODELS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the trainer.

        Args:
            config (Optional[Dict[str, Any]]): Model configurations and hyperparameters.
        """
        self.settings = get_settings()
        self.config = config or {}
        if not self.config and hasattr(self.settings, 'models'):
            self.config = self.settings.models

    def get_model_params(self, name: str) -> dict:
        """
        Get hyperparameters for a specific model from config.

        Args:
            name (str): The name of the model.

        Returns:
            dict: The hyperparameters.
        """
        if hasattr(self.config, name):
            model_config = getattr(self.config, name)
            if hasattr(model_config, 'model_dump'):
                return model_config.model_dump()
            elif isinstance(model_config, dict):
                return model_config
        elif isinstance(self.config, dict):
            return self.config.get(name, {})
        return {}

    def train_single(
        self, name: str, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> BaseEstimator:
        """
        Train a single model.

        Args:
            name (str): The name of the model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional kwargs to override model parameters.

        Returns:
            BaseEstimator: The trained model.
            
        Raises:
            ValueError: If the model name is not supported.
        """
        if name not in self.MODELS:
            raise ValueError(f"Model {name} is not supported. Supported models: {list(self.MODELS.keys())}")

        params = self.get_model_params(name)
        params.update(kwargs)

        if name == 'xgboost':
            spw = params.pop('scale_pos_weight', None)
            if spw is None or spw == 'auto':
                neg_count = int(np.sum(y_train == 0))
                pos_count = int(np.sum(y_train == 1))
                if pos_count > 0:
                    params['scale_pos_weight'] = neg_count / pos_count
                    logger.info(f"Auto-computed scale_pos_weight for xgboost: {params['scale_pos_weight']:.2f}")
            elif spw is not None:
                params['scale_pos_weight'] = spw

        model_cls = self.MODELS[name]
        
        logger.info(f"Training {name} with parameters: {params}")
        model = model_cls(**params)
        model.fit(X_train, y_train)
        logger.info(f"Finished training {name}.")
        
        return model

    def train_all(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, BaseEstimator]:
        """
        Train all configured models.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            Dict[str, BaseEstimator]: A dictionary of trained models.
        """
        trained_models = {}
        for name in self.MODELS.keys():
            try:
                trained_models[name] = self.train_single(name, X_train, y_train)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        return trained_models

    def cross_validate(
        self, name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation for a model.

        Args:
            name (str): The name of the model.
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.
            cv (int): Number of cross-validation folds.

        Returns:
            Dict[str, float]: Cross-validation metrics (mean scores).
        """
        if name not in self.MODELS:
            raise ValueError(f"Model {name} is not supported.")
            
        params = self.get_model_params(name)
        if name == 'xgboost':
            spw = params.pop('scale_pos_weight', None)
            if spw is None or spw == 'auto':
                neg = int(np.sum(y == 0))
                pos = int(np.sum(y == 1))
                if pos > 0:
                    params['scale_pos_weight'] = neg / pos
            elif spw is not None:
                params['scale_pos_weight'] = spw

        model = self.MODELS[name](**params)
        logger.info(f"Starting {cv}-fold cross-validation for {name}")
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        metrics = {
            f"mean_cv_{metric}": float(np.mean(cv_results[f"test_{metric}"]))
            for metric in scoring
        }
        
        logger.info(f"Cross-validation metrics for {name}: {metrics}")
        return metrics
