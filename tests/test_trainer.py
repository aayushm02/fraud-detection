import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def synthetic_data():
    """Generate small synthetic data for fast training."""
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n)
    })
    y = np.random.choice([0, 1], n, p=[0.9, 0.1])
    return X, y

def test_model_training(synthetic_data):
    """Test model trains without error."""
    X, y = synthetic_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    assert preds.shape == (1000,)
    assert probs.shape == (1000,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

def test_model_save_load(synthetic_data, tmp_path):
    """Test model saving and loading."""
    import joblib
    import os
    
    X, y = synthetic_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    
    assert os.path.exists(model_path)
    
    loaded_model = joblib.load(model_path)
    preds_original = model.predict(X)
    preds_loaded = loaded_model.predict(X)
    
    np.testing.assert_array_equal(preds_original, preds_loaded)
