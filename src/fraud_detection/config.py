import os
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

class ProjectConfig(BaseModel):
    name: str
    version: str

class DataConfig(BaseModel):
    credit_card_path: str
    synthetic_banking_size: int
    test_size: float
    random_state: int

class FeaturesConfig(BaseModel):
    scaling: str
    handle_imbalance: str
    velocity_windows: List[int]
    outlier_clip_factor: float

from typing import Union

class LogisticRegressionConfig(BaseModel):
    C: float
    max_iter: int
    class_weight: str

class RandomForestConfig(BaseModel):
    n_estimators: int
    max_depth: int
    class_weight: str
    n_jobs: int

class XGBoostConfig(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    scale_pos_weight: Union[str, float, int]
    eval_metric: str
    tree_method: str

class LightGBMConfig(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    is_unbalance: bool
    verbose: int

class ModelsConfig(BaseModel):
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig

class EvaluationConfig(BaseModel):
    threshold: float
    metrics: List[str]
    cost_false_positive: float
    cost_false_negative: float

class ApiConfig(BaseModel):
    host: str
    port: int

class DashboardConfig(BaseModel):
    port: int

class RegistryConfig(BaseModel):
    path: str

class Settings(BaseSettings):
    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    api: ApiConfig
    dashboard: DashboardConfig
    registry: RegistryConfig

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_DETECTION_",
        env_nested_delimiter="__",
    )

_settings_instance = None

def _find_config_path(config_path: str) -> str:
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return config_path
    # Try relative to CWD
    if os.path.exists(config_path):
        return config_path
    # Try relative to project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(project_root, config_path)
    if os.path.exists(candidate):
        return candidate
    return config_path  # fallback

def get_settings(config_path: str = "config/config.yaml", force_reload: bool = False) -> Settings:
    global _settings_instance
    if _settings_instance is None or force_reload:
        actual_path = _find_config_path(config_path)
        with open(actual_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        _settings_instance = Settings(**yaml_data)
    return _settings_instance
