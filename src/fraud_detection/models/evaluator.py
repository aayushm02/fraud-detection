import os
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
from matplotlib.figure import Figure

from fraud_detection.config import get_settings

logger = logging.getLogger(__name__)

class FraudModelEvaluator:
    """Evaluates machine learning models for fraud detection."""

    def __init__(self, models: Dict[str, BaseEstimator], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Initialize the evaluator.

        Args:
            models (Dict[str, BaseEstimator]): Dictionary of trained models.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.settings = get_settings()

    def evaluate_single(self, name: str) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            name (str): The name of the model to evaluate.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if name not in self.models:
            raise ValueError(f"Model {name} is not in the models dictionary.")

        model = self.models[name]
        y_pred = model.predict(self.X_test)
        
        # Predict proba for positive class if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.X_test)[:, 1]
        else:
            y_prob = y_pred # Fallback if predict_proba is not available

        metrics = {
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(self.y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(self.y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(self.y_test, y_prob)),
            'average_precision': float(average_precision_score(self.y_test, y_prob))
        }

        logger.info(f"Evaluation metrics for {name}: {metrics}")
        return metrics

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all models and return a summary dataframe.

        Returns:
            pd.DataFrame: A dataframe containing evaluation metrics for all models.
        """
        results = {}
        for name in self.models.keys():
            try:
                results[name] = self.evaluate_single(name)
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        df = pd.DataFrame(results).T
        return df

    def _get_probs(self, name: str) -> np.ndarray:
        model = self.models[name]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(self.X_test)[:, 1]
        return model.predict(self.X_test)

    def plot_roc_curves(self, save_path: str = None) -> Figure:
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name in self.models.keys():
            y_prob = self._get_probs(name)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_val = roc_auc_score(self.y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
            
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        sns.despine()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves plot to {save_path}")
            
        return fig

    def plot_pr_curves(self, save_path: str = None) -> Figure:
        """Plot Precision-Recall curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name in self.models.keys():
            y_prob = self._get_probs(name)
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            ap_val = average_precision_score(self.y_test, y_prob)
            ax.plot(recall, precision, label=f"{name} (AP = {ap_val:.3f})")
            
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        sns.despine()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curves plot to {save_path}")
            
        return fig

    def plot_confusion_matrix(self, name: str, save_path: str = None) -> Figure:
        """Plot confusion matrix for a specific model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")

        y_pred = self.models[name].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {name}')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix for {name} to {save_path}")

        return fig

    def plot_all_confusion_matrices(self, save_path: str = None) -> Figure:
        """Plot confusion matrices for all models in a grid."""
        n_models = len(self.models)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
        axes = axes.flatten()

        for i, name in enumerate(self.models.keys()):
            y_pred = self.models[name].predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
            axes[i].set_title(f'{name}')

        # Hide any empty subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved all confusion matrices to {save_path}")

        return fig

    def find_optimal_threshold(self, name: str, metric: str = 'f1') -> float:
        """Find the optimal threshold for classifying predictions."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")
            
        y_prob = self._get_probs(name)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_prob)
        
        if metric == 'f1':
            # Calculate F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        else:
            raise ValueError("Only 'f1' metric is currently supported for threshold optimization.")
            
        return float(optimal_threshold)

    def cost_benefit_analysis(self, name: str, cost_fp: float = 10.0, cost_fn: float = 500.0) -> Dict[str, float]:
        """Perform a cost-benefit analysis based on confusion matrix."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")

        y_pred = self.models[name].predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

        total_cost = (fp * cost_fp) + (fn * cost_fn)
        savings = (tp * cost_fn) - (fp * cost_fp)
        
        analysis = {
            'total_cost': float(total_cost),
            'savings': float(savings),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return analysis

    def generate_report(self, output_dir: str) -> str:
        """Generate a markdown report summarizing model evaluations."""
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.evaluate_all()
        
        roc_path = os.path.join(output_dir, 'roc_curves.png')
        self.plot_roc_curves(roc_path)
        
        pr_path = os.path.join(output_dir, 'pr_curves.png')
        self.plot_pr_curves(pr_path)
        
        cm_path = os.path.join(output_dir, 'confusion_matrices.png')
        self.plot_all_confusion_matrices(cm_path)
        
        md_content = f"# Model Evaluation Report\n\n"
        md_content += "## Metrics Summary\n"
        md_content += df.to_markdown() + "\n\n"
        
        md_content += "## Cost-Benefit Analysis\n"
        for name in self.models.keys():
            cba = self.cost_benefit_analysis(name)
            md_content += f"### {name}\n"
            md_content += f"- Total Cost: ${cba['total_cost']:.2f}\n"
            md_content += f"- Total Savings: ${cba['savings']:.2f}\n\n"
            
        md_content += "## ROC Curves\n"
        md_content += f"![ROC Curves](roc_curves.png)\n\n"
        
        md_content += "## Precision-Recall Curves\n"
        md_content += f"![PR Curves](pr_curves.png)\n\n"
        
        md_content += "## Confusion Matrices\n"
        md_content += f"![Confusion Matrices](confusion_matrices.png)\n"
        
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(md_content)
            
        logger.info(f"Generated evaluation report at {report_path}")
        return report_path
