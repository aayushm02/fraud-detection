"""Feature engineering module."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from fraud_detection.config import get_settings

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates features for both credit card and banking datasets."""
    
    def __init__(self):
        settings = get_settings()
        self.velocity_windows = settings.features.velocity_windows
        self.feature_names_out = []
        self.amount_mean_ = 0.0
        self.amount_std_ = 1.0
        self.global_median_ = 0.0
        self.account_avg_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureEngineer":
        if "amount" in X.columns:
            self.amount_mean_ = X["amount"].mean()
            self.amount_std_ = X["amount"].std()
            if pd.isna(self.amount_std_) or self.amount_std_ == 0:
                self.amount_std_ = 1.0
            self.global_median_ = X["amount"].median()
            
            if "sender_account" in X.columns:
                self.account_avg_ = X.groupby("sender_account")["amount"].mean().to_dict()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        original_index = df.index.copy()
        
        has_time = "timestamp" in df.columns
        has_amount = "amount" in df.columns
        has_account = "sender_account" in df.columns
        has_temporal = has_time and has_account
        
        if has_time:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
            
        if has_temporal:
            df['_orig_idx'] = original_index
            
            if has_amount:
                # Sort by account + timestamp, then use groupby + rolling count
                df = df.sort_values(['sender_account', 'timestamp'])
                df = df.set_index('timestamp')
                for window in self.velocity_windows:
                    df[f'velocity_{window}h'] = df.groupby('sender_account')['amount'].transform(
                        lambda x: x.rolling(f'{window}h', min_periods=1).count()
                    )
                df = df.reset_index()
                
                df["time_since_last"] = df.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)
                df["amount_x_time"] = df["amount"] * df["time_since_last"]
                
                if self.velocity_windows and f"velocity_{self.velocity_windows[0]}h" in df.columns:
                    df["amount_x_velocity"] = df["amount"] * df[f"velocity_{self.velocity_windows[0]}h"]
                    
                df["amount_rolling_std_10"] = df.groupby("sender_account")["amount"].transform(
                    lambda x: x.rolling(10, min_periods=1).std()
                ).fillna(0)
                
            df.index = df['_orig_idx']
            df = df.loc[original_index]
            df = df.drop(columns=['_orig_idx'])
            
        if has_amount:
            df["amount_vs_global_median"] = df["amount"] / (self.global_median_ + 1e-6)
            df["amount_zscore"] = (df["amount"] - self.amount_mean_) / (self.amount_std_ + 1e-6)
            
            if has_account:
                account_avg = df["sender_account"].map(self.account_avg_).fillna(self.amount_mean_)
                df["amount_vs_account_avg"] = df["amount"] / (account_avg + 1e-6)
                
        if has_time:
            df = df.drop(columns=["timestamp"])
            
        self.feature_names_out = df.columns.tolist()
        return df
        
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out)
