#!/usr/bin/env python3
"""
Modern training script for serious event prediction using the new ML pipeline framework.
This script demonstrates how to use the production-ready ML pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import MLTrainer
from utils.logger import get_logger
from config.config_manager import config_manager

logger = get_logger(__name__)


def main():
    """Main training function."""
    logger.info("Starting serious event prediction training")
    
    try:
        # Initialize trainer
        trainer = MLTrainer(task_name="serious_prediction")
        
        # Run full training pipeline
        results = trainer.run_full_pipeline(
            table_name="openfda_events",
            optimize_hyperparams=True  # Enable hyperparameter optimization
        )
        
        # Print results summary
        print_training_results(results)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def print_training_results(results):
    """Print training results in a formatted way."""
    print("\n" + "="*60)
    print("SERIOUS EVENT PREDICTION TRAINING RESULTS")
    print("="*60)
    
    # Task information
    print(f"Task: {results['task_name']}")
    print(f"Training samples: {results['data_info']['train_samples']:,}")
    print(f"Test samples: {results['data_info']['test_samples']:,}")
    print(f"Features: {results['data_info']['features']}")
    
    # Summary
    summary = results['summary']
    print(f"\nModels trained: {summary['successful_models']}/{summary['total_models']}")
    
    if 'best_model' in summary and summary['best_model']:
        print(f"Best model: {summary['best_model']}")
        print(f"Best accuracy: {summary['best_accuracy']:.4f}")
    
    # Individual model results
    print(f"\nModel Performance:")
    print("-" * 40)
    
    for model_name, result in results['results'].items():
        if 'evaluation' in result:
            metrics = result['evaluation']['metrics']
            print(f"{model_name:20} | Accuracy: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1_macro']:.4f} | "
                  f"Precision: {metrics['precision_macro']:.4f}")
        else:
            print(f"{model_name:20} | Failed: {result.get('error', 'Unknown error')}")
    
    # MLflow information
    print(f"\nMLflow Integration:")
    print("-" * 40)
    config = config_manager.load_config()
    print(f"Tracking URI: {config.mlflow.tracking_uri}")
    print(f"Experiment: {config.mlflow.experiment_name}")
    print(f"Model Registry: {config.mlflow.model_registry_name}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

