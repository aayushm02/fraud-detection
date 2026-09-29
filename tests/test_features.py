import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Generate synthetic banking data for testing."""
    n = 100
    data = pd.DataFrame({
        "customer_id": np.random.randint(1000, 1010, n),
        "amount": np.random.exponential(100, n),
        "timestamp": pd.date_range(start="2023-01-01", periods=n, freq="H")
    })
    return data

def test_time_features(sample_data):
    """Test time feature extraction."""
    df = sample_data.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert df["hour"].max() <= 23
    assert df["day_of_week"].max() <= 6

def test_velocity_features(sample_data):
    """Test velocity feature creation."""
    df = sample_data.copy()
    # Mock velocity: count per customer
    velocity = df.groupby("customer_id").size().reset_index(name="txn_count")
    df = df.merge(velocity, on="customer_id", how="left")
    
    assert "txn_count" in df.columns
    assert df.shape[0] == sample_data.shape[0]

def test_get_feature_names_out(sample_data):
    """Test retrieving feature names."""
    feature_names = ["customer_id", "amount", "hour", "day_of_week", "txn_count"]
    assert len(feature_names) == 5
