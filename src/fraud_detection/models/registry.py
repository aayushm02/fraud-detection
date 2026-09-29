import os
import json
import logging
from datetime import datetime
import joblib
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import BaseEstimator

from fraud_detection.config import get_settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages model versioning, saving, and loading."""

    def __init__(self, registry_path: str = 'models/') -> None:
        """
        Initialize the model registry.

        Args:
            registry_path (str): Base directory for saving models.
        """
        self.settings = get_settings()
        # Fallback to configured path if absolute path is not given
        if not os.path.isabs(registry_path):
            base_dir = getattr(self.settings, 'base_dir', '.')
            self.registry_path = os.path.join(base_dir, registry_path)
        else:
            self.registry_path = registry_path
            
        os.makedirs(self.registry_path, exist_ok=True)
        self.metadata_file = os.path.join(self.registry_path, 'registry.json')
        
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump([], f)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: List[Dict[str, Any]]) -> None:
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def save(self, model: BaseEstimator, name: str, metrics: Dict[str, float], version: str = 'auto', config_hash: str = '', feature_names: List[str] = None) -> str:
        """
        Save a model and its metadata.

        Args:
            model (BaseEstimator): The trained model object.
            name (str): Name of the model.
            metrics (Dict[str, float]): Performance metrics.
            version (str): Version identifier. If 'auto', automatically increments (v1, v2, ...).
            config_hash (str): Hash of the training configuration.
            feature_names (List[str]): List of feature names used for training.

        Returns:
            str: The version that was saved.
        """
        registry_data = self._load_metadata()
        
        # Get existing versions for this model
        existing = [entry for entry in registry_data if entry['name'] == name]
        
        if version == 'auto':
            if not existing:
                version = 'v1'
            else:
                versions = [int(e['version'].replace('v', '')) for e in existing if e['version'].startswith('v') and e['version'][1:].isdigit()]
                next_v = max(versions) + 1 if versions else 1
                version = f"v{next_v}"
        
        model_filename = f"{name}_{version}.joblib"
        model_path = os.path.join(self.registry_path, model_filename)
        
        # Save model object
        joblib.dump(model, model_path)
        
        # Prepare metadata
        meta_entry = {
            'name': name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'config_hash': config_hash,
            'feature_names': feature_names or []
        }
        
        # Update registry.json
        # Remove old entry if same version is overwritten
        registry_data = [entry for entry in registry_data if not (entry['name'] == name and entry['version'] == version)]
        registry_data.append(meta_entry)
        
        self._save_metadata(registry_data)
        logger.info(f"Saved model {name} version {version} at {model_path}")
        
        return version

    def load(self, name: str, version: str = 'latest') -> BaseEstimator:
        """
        Load a model by name and version.

        Args:
            name (str): Model name.
            version (str): Version to load. Defaults to 'latest'.

        Returns:
            BaseEstimator: The loaded model.
        """
        registry_data = self._load_metadata()
        existing = [entry for entry in registry_data if entry['name'] == name]
        
        if not existing:
            raise ValueError(f"No models found for name {name}.")
            
        if version == 'latest':
            # Sort by timestamp to find latest
            existing.sort(key=lambda x: x['timestamp'], reverse=True)
            target = existing[0]
        else:
            targets = [e for e in existing if e['version'] == version]
            if not targets:
                raise ValueError(f"Version {version} not found for model {name}.")
            target = targets[0]
            
        model_path = target['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Loaded model {name} version {target['version']}")
        return joblib.load(model_path)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models and their metadata."""
        return self._load_metadata()

    def get_best(self, metric: str = 'f1') -> Optional[Dict[str, Any]]:
        """
        Get the best model across all versions based on a specific metric.

        Args:
            metric (str): The metric to optimize.

        Returns:
            Optional[Dict[str, Any]]: Model dictionary with model object and metadata.
        """
        registry_data = self._load_metadata()
        if not registry_data:
            return None
            
        valid_entries = [e for e in registry_data if metric in e.get('metrics', {})]
        if not valid_entries:
            return None
            
        best_entry = max(valid_entries, key=lambda x: x['metrics'][metric])
        model = self.load(best_entry['name'], best_entry.get('version', 'latest'))
        
        logger.info(f"Best model based on {metric} is {best_entry['name']} version {best_entry['version']}")
        return {
            'name': best_entry['name'],
            'model': model,
            'metrics': best_entry.get('metrics', {}),
            'version': best_entry.get('version', 'unknown'),
            'timestamp': best_entry.get('timestamp', 'unknown'),
            'feature_names': best_entry.get('feature_names', []),
        }

    def delete(self, name: str, version: str) -> bool:
        """
        Delete a model version from the registry and filesystem.

        Args:
            name (str): Model name.
            version (str): Version to delete.

        Returns:
            bool: True if deleted successfully.
        """
        registry_data = self._load_metadata()
        targets = [e for e in registry_data if e['name'] == name and e['version'] == version]
        
        if not targets:
            logger.warning(f"Model {name} version {version} not found.")
            return False
            
        target = targets[0]
        try:
            if os.path.exists(target['model_path']):
                os.remove(target['model_path'])
        except Exception as e:
            logger.error(f"Error deleting model file: {e}")
            return False
            
        registry_data = [e for e in registry_data if not (e['name'] == name and e['version'] == version)]
        self._save_metadata(registry_data)
        
        logger.info(f"Deleted model {name} version {version}")
        return True
