"""Preprocessing pipeline for fraud detection."""
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from fraud_detection.config import get_settings

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline for fraud detection."""
    
    def __init__(self):
        settings = get_settings()
        self.scaling = settings.features.scaling
        self.handle_imbalance = settings.features.handle_imbalance
        self.clip_factor = settings.features.outlier_clip_factor
        
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.label_encoders = {}
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
            
        self.numeric_cols = []
        self.categorical_cols = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudPreprocessor":
        X = X.copy()
        if "timestamp" in X.columns:
            X["timestamp"] = pd.to_datetime(X["timestamp"]).astype('int64') // 10**9
            
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        for col in self.numeric_cols:
            self.numeric_medians[col] = X[col].median()
            
        for col in self.categorical_cols:
            self.categorical_modes[col] = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna(self.categorical_modes[col]))
            self.label_encoders[col] = le
            
        # Fit scaler
        X_temp = X.copy()
        for col in self.numeric_cols:
            X_temp[col] = X_temp[col].fillna(self.numeric_medians[col])
            
            # Outlier clipping
            Q1 = X_temp[col].quantile(0.25)
            Q3 = X_temp[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.clip_factor * IQR
            self.upper_bounds_[col] = Q3 + self.clip_factor * IQR
            X_temp[col] = X_temp[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
            
        if self.scaler and self.numeric_cols:
            self.scaler.fit(X_temp[self.numeric_cols])
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if "timestamp" in X_out.columns:
            X_out["timestamp"] = pd.to_datetime(X_out["timestamp"]).astype('int64') // 10**9
            
        # Missing values
        for col in self.numeric_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.numeric_medians.get(col, 0))
                
        for col in self.categorical_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.categorical_modes.get(col, "Unknown"))
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    col_str = X_out[col].astype(str)
                    known = set(le.classes_)
                    col_str = col_str.apply(lambda x: x if x in known else le.classes_[0])
                    X_out[col] = le.transform(col_str)
                    
        # Outlier clipping and Scaling
        for col in self.numeric_cols:
            if col in X_out.columns:
                lower_bound = self.lower_bounds_.get(col, X_out[col].min())
                upper_bound = self.upper_bounds_.get(col, X_out[col].max())
                X_out[col] = X_out[col].clip(lower=lower_bound, upper=upper_bound)
                
        if self.scaler and self.numeric_cols:
            cols_to_scale = [c for c in self.numeric_cols if c in X_out.columns]
            if cols_to_scale:
                X_out[cols_to_scale] = self.scaler.transform(X_out[cols_to_scale])
                
        return X_out
        
    def save(self, filepath: str) -> None:
        """Save fitted preprocessor to disk."""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "FraudPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(filepath)

def prepare_data(X: pd.DataFrame, y: pd.Series, preprocessor: FraudPreprocessor) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply train-test split and SMOTE."""
    settings = get_settings()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.data.test_size, stratify=y, random_state=settings.data.random_state
    )
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    if settings.features.handle_imbalance == "smote":
        smote = SMOTE(random_state=settings.data.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test
