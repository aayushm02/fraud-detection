"""
Models subpackage for fraud detection.

Imports are lazy to avoid loading heavy ML libraries (sklearn, xgboost, lightgbm)
at module import time — this keeps the API startup fast and avoids DLL issues.
"""


def __getattr__(name: str):
    if name == "FraudModelTrainer":
        from fraud_detection.models.trainer import FraudModelTrainer
        return FraudModelTrainer
    elif name == "FraudModelEvaluator":
        from fraud_detection.models.evaluator import FraudModelEvaluator
        return FraudModelEvaluator
    elif name == "ModelRegistry":
        from fraud_detection.models.registry import ModelRegistry
        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FraudModelTrainer", "FraudModelEvaluator", "ModelRegistry"]
