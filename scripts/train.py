"""Train fraud detection models.

Usage:
    python scripts/train.py                    # Train all models on synthetic banking data
    python scripts/train.py --model xgboost    # Train single model
    python scripts/train.py --dataset both     # Train on both datasets
"""
import argparse
import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_detection.config import get_settings
from fraud_detection.data.loader import DatasetLoader
from fraud_detection.data.preprocessor import FraudPreprocessor
from fraud_detection.features.engineer import FeatureEngineer
from fraud_detection.models.trainer import FraudModelTrainer
from fraud_detection.models.evaluator import FraudModelEvaluator
from fraud_detection.models.registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to train (logistic_regression, random_forest, xgboost, lightgbm)')
    parser.add_argument('--dataset', type=str, default='banking',
                       choices=['credit_card', 'banking', 'both'],
                       help='Dataset to use')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train')
    
    settings = get_settings(args.config)
    
    # Load data
    datasets = []
    if args.dataset in ('banking', 'both'):
        logger.info('Loading synthetic banking dataset...')
        df = DatasetLoader.load("banking", n_samples=settings.data.synthetic_banking_size)
        datasets.append(('banking', df))
    if args.dataset in ('credit_card', 'both'):
        logger.info('Loading credit card dataset...')
        try:
            df = DatasetLoader.load("credit_card", path=settings.data.credit_card_path)
            datasets.append(('credit_card', df))
        except FileNotFoundError:
            logger.warning('Credit card dataset not found. Download it first. See data/README.md')
    
    if not datasets:
        logger.error('No datasets loaded. Exiting.')
        sys.exit(1)
    
    registry = ModelRegistry()
    
    for dataset_name, df in datasets:
        logger.info(f'\n{"="*60}')
        logger.info(f'Training on {dataset_name} dataset ({len(df)} rows)')
        logger.info(f'{"="*60}')
        
        # Separate target
        target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Feature engineering
        engineer = FeatureEngineer()
        engineer.fit(X)
        X = engineer.transform(X)
        
        # Preprocessing
        preprocessor = FraudPreprocessor()
        X, y = preprocessor.fit_transform(X, y)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.data.test_size,
            random_state=settings.data.random_state, stratify=y
        )
        
        # Train
        trainer = FraudModelTrainer()
        if args.model:
            models = {args.model: trainer.train_single(args.model, X_train, y_train)}
        else:
            models = trainer.train_all(X_train, y_train)
        
        # Evaluate
        evaluator = FraudModelEvaluator(models, X_test, y_test)
        results = evaluator.evaluate_all()
        
        print(f'\n{"="*60}')
        print(f'Results for {dataset_name}')
        print(f'{"="*60}')
        print(results.to_string())
        
        # Save to registry
        for name, model in models.items():
            metrics = evaluator.evaluate_single(name)
            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else []
            registry.save(model, f'{dataset_name}_{name}', metrics, feature_names=feature_names)
            logger.info(f'Saved {dataset_name}_{name} to registry')
    
    print('\nTraining complete! Models saved to registry.')


if __name__ == '__main__':
    main()
