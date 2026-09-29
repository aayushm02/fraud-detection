"""
Explainability subpackage for fraud detection.
"""

def __getattr__(name: str):
    if name == "FraudExplainer":
        from fraud_detection.explainability.shap_explainer import FraudExplainer
        return FraudExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["FraudExplainer"]
