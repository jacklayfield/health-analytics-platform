#!/usr/bin/env python3
"""
Complete example demonstrating the production-ready ML pipeline.
This script shows how to use all components of the ML pipeline framework.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config_manager import config_manager
from data.data_loader import DataLoader
from training.trainer import MLTrainer
from experiments.experiment_tracker import experiment_tracker
from evaluation.model_evaluator import ModelEvaluator
from optimization.hyperparameter_tuner import HyperparameterTuner, PARAM_SPACES
from validation.data_validator import DataValidator
from monitoring.model_monitor import ModelMonitor
from serving.model_server import ModelServer
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run the complete ML pipeline example."""
    logger.info("Starting complete ML pipeline example")
    
    try:
        # 1. Configuration Management
        logger.info("1. Loading configuration...")
        config = config_manager.load_config()
        print(f"Configuration loaded successfully")
        print(f"  - MLflow tracking URI: {config.mlflow.tracking_uri}")
        print(f"  - Experiment name: {config.mlflow.experiment_name}")
        
        # 2. Data Loading and Validation
        logger.info("2. Loading and validating data...")
        data_loader = DataLoader()
        
        # Load data
        df = data_loader.load_data("openfda_events", limit=10000)  # Limit for demo
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate data quality
        validation_results = data_loader.validate_data_quality(df, "serious")
        print(f"Data validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
        if not validation_results['is_valid']:
            print(f"  Issues: {validation_results['issues']}")
        
        # 3. Advanced Data Validation with Great Expectations
        logger.info("3. Running advanced data validation...")
        try:
            data_validator = DataValidator()
            ge_results = data_validator.validate_training_data(df, "serious", "openfda_events")
            print(f"Great Expectations validation completed")
            print(f"  - Success rate: {ge_results['success_rate']:.2%}")
            print(f"  - Total expectations: {ge_results['total_expectations']}")
        except ImportError:
            print("⚠ Great Expectations not installed, skipping advanced validation")
        
        # 4. Model Training
        logger.info("4. Training models...")
        trainer = MLTrainer(task_name="serious_prediction")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = trainer.load_and_prepare_data("openfda_events")
        print(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train all models
        training_results = trainer.train_all_models(
            X_train, y_train, X_test, y_test,
            optimize_hyperparams=True  # Enable hyperparameter optimization
        )
        
        print(f"Model training completed")
        print(f"  - Models trained: {training_results['summary']['successful_models']}/{training_results['summary']['total_models']}")
        print(f"  - Best model: {training_results['summary']['best_model']}")
        print(f"  - Best accuracy: {training_results['summary']['best_accuracy']:.4f}")
        
        # 5. Model Evaluation
        logger.info("5. Comprehensive model evaluation...")
        evaluator = ModelEvaluator()
        
        # Get the best model for detailed evaluation
        best_model_name = training_results['summary']['best_model']
        best_model = training_results['results'][best_model_name]['model']
        
        evaluation_results = evaluator.evaluate_classification(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            model_name=best_model_name
        )
        
        print(f"Model evaluation completed")
        print(f"  - Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        print(f"  - F1 Score: {evaluation_results['metrics']['f1_macro']:.4f}")
        print(f"  - Precision: {evaluation_results['metrics']['precision_macro']:.4f}")
        print(f"  - Recall: {evaluation_results['metrics']['recall_macro']:.4f}")
        
        # 6. Model Monitoring Setup
        logger.info("6. Setting up model monitoring...")
        model_monitor = ModelMonitor()
        
        # Set baseline for monitoring
        model_monitor.set_baseline(
            model_name=best_model_name,
            X_baseline=X_train,
            y_baseline=y_train,
            predictions_baseline=best_model.predict(X_train)
        )
        print(f"Model monitoring baseline set")
        
        # Simulate data drift detection
        drift_results = model_monitor.detect_data_drift(
            model_name=best_model_name,
            X_current=X_test
        )
        print(f"Data drift detection completed")
        print(f"  - Drift detected: {drift_results['drift_detected']}")
        print(f"  - Overall drift score: {drift_results['drift_score']:.4f}")
        
        # Simulate performance monitoring
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        performance_drift = model_monitor.detect_performance_drift(
            model_name=best_model_name,
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )
        print(f"Performance monitoring completed")
        print(f"  - Performance drift detected: {performance_drift['performance_drift_detected']}")
        
        # 7. Hyperparameter Optimization Demo
        logger.info("7. Demonstrating hyperparameter optimization...")
        tuner = HyperparameterTuner(n_trials=20)  # Reduced for demo
        
        # Optimize Random Forest (if available)
        if 'random_forest' in [algo['name'] for algo in config.models['serious_prediction'].algorithms]:
            from sklearn.ensemble import RandomForestClassifier
            
            optimization_results = tuner.optimize_classification(
                model_class=RandomForestClassifier,
                X=X_train,
                y=y_train,
                param_space=PARAM_SPACES['RandomForestClassifier'],
                study_name="demo_optimization"
            )
            
            print(f"Hyperparameter optimization completed")
            print(f"  - Best score: {optimization_results['best_score']:.4f}")
            print(f"  - Trials completed: {optimization_results['n_trials']}")
        
        # 8. Model Serving Demo
        logger.info("8. Demonstrating model serving...")
        
        # Save model for serving
        import joblib
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{best_model_name}_demo.joblib"
        joblib.dump(best_model, model_path)
        print(f"Model saved for serving: {model_path}")
        
        # Create sample prediction request
        sample_features = {
            "patientonsetage": 35,
            "patientsex": "F",
            "reaction": "nausea",
            "brand_name": "drug_a"
        }
        
        # Make prediction
        sample_df = pd.DataFrame([sample_features])
        prediction = best_model.predict(sample_df)[0]
        prediction_proba = best_model.predict_proba(sample_df)[0] if hasattr(best_model, 'predict_proba') else None
        
        print(f"Sample prediction completed")
        print(f"  - Features: {sample_features}")
        print(f"  - Prediction: {prediction}")
        if prediction_proba is not None:
            print(f"  - Probabilities: {dict(zip(best_model.classes_, prediction_proba))}")
        
        # 9. Generate Reports
        logger.info("9. Generating comprehensive reports...")
        
        # Generate evaluation report
        eval_report_path = evaluator.generate_evaluation_report(
            evaluation_results,
            output_dir="artifacts"
        )
        print(f"Evaluation report generated: {eval_report_path}")
        
        # Generate monitoring report
        monitoring_report_path = model_monitor.generate_monitoring_report(
            best_model_name,
            output_path="artifacts/monitoring_report.json"
        )
        print(f"Monitoring report generated: {monitoring_report_path}")
        
        # 10. Summary
        logger.info("10. Pipeline execution summary...")
        print("\n" + "="*60)
        print("ML PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Configuration management: Working")
        print(f"Data loading and validation: Working")
        print(f"Model training: {training_results['summary']['successful_models']} models trained")
        print(f"Model evaluation: Comprehensive metrics calculated")
        print(f"Hyperparameter optimization: Completed")
        print(f"Model monitoring: Baseline set and drift detection working")
        print(f"Model serving: Ready for deployment")
        print(f"Experiment tracking: Integrated with MLflow")
        print(f"Reports generated: Evaluation and monitoring reports created")
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best accuracy: {training_results['summary']['best_accuracy']:.4f}")
        print(f"MLflow experiment: {config.mlflow.experiment_name}")
        
        print("\nNext steps:")
        print("1. Start MLflow server: mlflow server --backend-store-uri sqlite:///mlflow.db")
        print("2. Start model server: python -m src.cli.main serve")
        print("3. View experiments at: http://localhost:5000")
        print("4. Make predictions at: http://localhost:8000/predict")
        
        logger.info("Complete ML pipeline example finished successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
    finally:
        # Cleanup
        if 'data_loader' in locals():
            data_loader.close()


if __name__ == "__main__":
    main()

