import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning(f"SHAP import unavailable: {e}")

class FraudExplainer:
    """Provides SHAP-based explainability for fraud detection models."""

    def __init__(self, model: Any, X_train: Optional[pd.DataFrame] = None, feature_names: Optional[List[str]] = None) -> None:
        """
        Initialize the explainer.

        Args:
            model (Any): The trained machine learning model.
            X_train (Optional[pd.DataFrame]): Training data used as background dataset.
            feature_names (Optional[List[str]]): List of feature names.
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.is_tree = False
        
        if X_train is not None:
            if not self.feature_names:
                self.feature_names = list(X_train.columns)
            self._init_explainer()
        else:
            logger.warning("X_train is None. SHAP explainer not initialized.")
            if not self.feature_names:
                self.feature_names = []

    def _init_explainer(self) -> None:
        if not SHAP_AVAILABLE or shap is None:
            logger.warning("SHAP library is not available. Explainer disabled.")
            return

        tree_models = (
            'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
            'DecisionTreeClassifier', 'GradientBoostingClassifier'
        )
        
        model_name = type(self.model).__name__
        logger.info(f"Initializing SHAP explainer for {model_name}")
        
        try:
            if model_name in tree_models:
                self.explainer = shap.TreeExplainer(self.model)
                self.is_tree = True
            else:
                logger.warning("Using KernelExplainer. This might be slow for large datasets.")
                background = shap.kmeans(self.X_train, 100) if len(self.X_train) > 100 else self.X_train
                pred_func = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
                self.explainer = shap.KernelExplainer(pred_func, background)
                self.is_tree = False
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a given dataset.

        Args:
            X (pd.DataFrame): Features to compute SHAP values for.

        Returns:
            np.ndarray: SHAP values.
        """
        logger.info("Computing SHAP values...")
        shap_values = self.explainer.shap_values(X)
        
        # Handle cases where shap_values is a list (e.g., multi-class or some tree models outputting both classes)
        if isinstance(shap_values, list):
            # For binary classification, typically we want the explanation for the positive class (index 1)
            shap_values = shap_values[1]
            
        return shap_values

    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
        """
        Generate a SHAP summary plot (beeswarm).

        Args:
            X (pd.DataFrame): Dataset to explain.
            save_path (Optional[str]): Path to save the figure.

        Returns:
            Figure: The matplotlib Figure object.
        """
        shap_values = self.compute_shap_values(X)
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {save_path}")
            
        return fig

    def plot_feature_importance(self, X: pd.DataFrame, top_n: int = 20, save_path: Optional[str] = None) -> Figure:
        """
        Generate a SHAP bar plot for overall feature importance.

        Args:
            X (pd.DataFrame): Dataset to explain.
            top_n (int): Number of top features to show.
            save_path (Optional[str]): Path to save the figure.

        Returns:
            Figure: The matplotlib Figure object.
        """
        shap_values = self.compute_shap_values(X)
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", feature_names=self.feature_names, max_display=top_n, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP feature importance plot to {save_path}")
            
        return fig

    def plot_dependence(self, feature: str, X: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
        """
        Generate a SHAP dependence plot for a specific feature.

        Args:
            feature (str): The feature name.
            X (pd.DataFrame): Dataset to explain.
            save_path (Optional[str]): Path to save the figure.

        Returns:
            Figure: The matplotlib Figure object.
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names.")
            
        shap_values = self.compute_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values, X, feature_names=self.feature_names, ax=ax, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP dependence plot for {feature} to {save_path}")
            
        return fig

    def explain_prediction(self, x: pd.Series, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP force plot data.

        Args:
            x (pd.Series): Single instance to explain.
            save_path (Optional[str]): Path to save the force plot as HTML/image.

        Returns:
            Dict: Explanation details including top contributing features and prediction.
        """
        x_df = pd.DataFrame([x])
        shap_values = self.compute_shap_values(x_df)
        
        # Get base value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[1] # positive class
        else:
            base_value = self.explainer.expected_value
            
        # Get actual prediction
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(x_df)[0, 1]
            prediction = self.model.predict(x_df)[0]
        else:
            prob = None
            prediction = self.model.predict(x_df)[0]

        # Extract feature contributions for this instance
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val = shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]
            contributions.append((feat, float(val)))
            
        # Sort by absolute contribution magnitude
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)

        if save_path:
            shap.force_plot(
                base_value, 
                shap_values[0] if len(shap_values.shape) > 1 else shap_values, 
                x_df, 
                matplotlib=True, 
                show=False
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved force plot to {save_path}")

        return {
            'prediction': float(prediction),
            'probability': float(prob) if prob is not None else None,
            'base_value': float(base_value),
            'top_contributions': contributions[:5]  # Return top 5 contributors
        }

    def get_top_features(self, X: Optional[pd.DataFrame] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the overall top contributing features based on mean absolute SHAP value, with fallback to model feature importances.
        """
        if self.explainer is not None and X is not None:
            try:
                shap_values = self.compute_shap_values(X)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importance = list(zip(self.feature_names, mean_abs_shap))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                return [{'feature': name, 'importance': float(importance)} for name, importance in feature_importance[:top_n]]
            except Exception as e:
                logger.warning(f"Error computing SHAP values: {e}")

        # Fallback to model feature_importances_ if available
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return [{'feature': name, 'importance': float(importance)} for name, importance in feature_importance[:top_n]]

        return []
