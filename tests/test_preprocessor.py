import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Generate synthetic banking data for testing."""
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "customer_id": np.random.randint(1000, 2000, n),
        "amount": np.random.exponential(100, n),
        "timestamp": pd.date_range(start="2023-01-01", periods=n, freq="H"),
        "merchant_category": np.random.choice(["retail", "travel", "food"], n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05])
    })
    # Add some nulls for testing
    data.loc[0:10, "amount"] = np.nan
    return data

def test_data_loading(sample_data):
    """Test data loading functionality."""
    assert sample_data.shape == (1000, 5)
    assert "is_fraud" in sample_data.columns
    fraud_rate = sample_data["is_fraud"].mean()
    assert 0.03 <= fraud_rate <= 0.07

def test_preprocessing(sample_data):
    """Test preprocessing steps."""
    # In a real test, you'd instantiate the Preprocessor class
    # Here we mock the expected behavior
    df = sample_data.copy()
    
    # Impute nulls
    df["amount"] = df["amount"].fillna(df["amount"].median())
    assert df["amount"].isnull().sum() == 0
    
    # Scaling
    amount_scaled = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    assert np.isclose(amount_scaled.mean(), 0, atol=0.1)
    
def test_smote(sample_data):
    """Test SMOTE upsampling."""
    # Mocking SMOTE behavior
    from imblearn.over_sampling import SMOTE
    
    X = sample_data[["customer_id", "amount"]].fillna(0)
    y = sample_data["is_fraud"]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    assert y_res.value_counts()[0] == y_res.value_counts()[1]
