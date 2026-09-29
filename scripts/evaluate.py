"""Evaluate trained fraud detection models.

Usage:
    python scripts/evaluate.py                 # Evaluate all models
    python scripts/evaluate.py --report        # Generate full report
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_detection.models.registry import ModelRegistry
from fraud_detection.data.loader import DatasetLoader
from fraud_detection.data.preprocessor import FraudPreprocessor
from fraud_detection.features.engineer import FeatureEngineer
from fraud_detection.models.evaluator import FraudModelEvaluator
from fraud_detection.config import get_settings


def main():
    parser = argparse.ArgumentParser(description='Evaluate fraud detection models')
    parser.add_argument('--model', type=str, default=None, help='Specific model to evaluate')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for reports')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('evaluate')
    
    registry = ModelRegistry()
    available = registry.list_models()
    
    if not available:
        logger.error('No trained models found. Run `python scripts/train.py` first.')
        sys.exit(1)
    
    print(f'Found {len(available)} trained model(s):')
    for m in available:
        print(f"  - {m.get('name', 'unknown')} (v{m.get('version', '?')})")
    
    # Load test data for evaluation
    settings = get_settings()
    df = DatasetLoader.load('banking', n_samples=settings.data.synthetic_banking_size)
    target_col = 'is_fraud' if 'is_fraud' in df.columns else 'Class'
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    engineer = FeatureEngineer()
    engineer.fit(X)
    X = engineer.transform(X)
    
    preprocessor = FraudPreprocessor()
    X, y = preprocessor.fit_transform(X, y)
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=settings.data.test_size,
        random_state=settings.data.random_state, stratify=y
    )
    
    # Load models from registry
    models = {}
    for entry in available:
        name = entry.get('name', '')
        if args.model and args.model not in name:
            continue
        try:
            model = registry.load(name)
            models[name] = model
        except Exception as e:
            logger.warning(f'Could not load {name}: {e}')
    
    if not models:
        logger.error('No models loaded successfully.')
        sys.exit(1)
    
    evaluator = FraudModelEvaluator(models, X_test, y_test)
    results = evaluator.evaluate_all()
    print('\nModel Evaluation Results:')
    print(results.to_string())
    
    if args.report:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = evaluator.generate_report(args.output_dir)
        print(f'\nReport saved to: {report_path}')


if __name__ == '__main__':
    main()
