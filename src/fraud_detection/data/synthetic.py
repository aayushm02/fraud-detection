"""Synthetic banking data generator."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from fraud_detection.config import get_settings

def generate_synthetic_banking_data(n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate realistic synthetic banking/AML transaction data."""
    if seed is None:
        seed = get_settings().data.random_state
    np.random.seed(seed)
    
    # Base generation
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(minutes=int(np.random.randint(0, 30 * 24 * 60))) for _ in range(n_samples)]
    timestamps.sort()
    
    transaction_ids = [f"TXN_{i:08d}" for i in range(n_samples)]
    amounts = np.random.lognormal(mean=4.0, sigma=1.0, size=n_samples)
    
    sender_accounts = [f"ACC_{np.random.randint(1000, 2000)}" for _ in range(n_samples)]
    receiver_accounts = [f"ACC_{np.random.randint(1000, 9999)}" for _ in range(n_samples)]
    
    transaction_types = np.random.choice(
        ["transfer", "payment", "deposit", "withdrawal"], 
        size=n_samples, 
        p=[0.4, 0.3, 0.15, 0.15]
    )
    
    is_international = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    sender_balances = amounts + np.random.uniform(0, 10000, size=n_samples)
    receiver_balances = np.random.uniform(100, 20000, size=n_samples)
    
    is_fraud = np.zeros(n_samples, dtype=int)
    
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "timestamp": timestamps,
        "amount": amounts,
        "sender_account": sender_accounts,
        "receiver_account": receiver_accounts,
        "transaction_type": transaction_types,
        "is_international": is_international,
        "sender_balance": sender_balances,
        "receiver_balance": receiver_balances,
        "is_fraud": is_fraud
    })
    
    # Inject fraud patterns (approx 5%)
    n_fraud = int(n_samples * 0.05)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    
    for idx in fraud_indices:
        pattern = np.random.choice(["rapid", "large", "night_cross", "round", "draining"])
        df.at[idx, "is_fraud"] = 1
        
        if pattern == "rapid":
            base_time = df.at[idx, "timestamp"]
            for offset in range(1, np.random.randint(3, 6)):
                rapid_idx = (idx + offset) % len(df)
                df.at[rapid_idx, "sender_account"] = df.at[idx, "sender_account"]
                df.at[rapid_idx, "timestamp"] = base_time + pd.Timedelta(minutes=np.random.randint(1, 10))
                df.at[rapid_idx, "is_fraud"] = 1
        elif pattern == "large":
            df.at[idx, "amount"] = df["amount"].mean() + 4 * df["amount"].std()
        elif pattern == "night_cross":
            df.at[idx, "is_international"] = 1
            night_time = base_time + timedelta(
                days=int(np.random.randint(0, 30)), 
                hours=int(np.random.randint(0, 5))
            )
            df.at[idx, "timestamp"] = night_time
        elif pattern == "round":
            df.at[idx, "amount"] = np.random.choice([5000, 10000, 50000])
        elif pattern == "draining":
            df.at[idx, "amount"] = df.at[idx, "sender_balance"] * np.random.uniform(0.85, 0.99)
            
    return df
