#!/usr/bin/env python3
"""
Quick test script to verify the ML pipeline setup works.
This script creates mock data and tests the core functionality.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config.config_manager import config_manager
        print("Config manager imported")
        
        from utils.logger import get_logger
        print("Logger imported")
        
        from training.trainer import MLTrainer
        print("Trainer imported")
        
        from experiments.experiment_tracker import experiment_tracker
        print("Experiment tracker imported")
        
        from evaluation.model_evaluator import ModelEvaluator
        print("Model evaluator imported")
        
        from optimization.hyperparameter_tuner import HyperparameterTuner
        print("Hyperparameter tuner imported")
        
        from serving.model_server import ModelServer
        print("Model server imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def create_mock_data():
    """Create mock data for testing."""
    print("\nCreating mock data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic mock data
    data = {
        'patientonsetage': np.random.normal(45, 20, n_samples).astype(int),
        'patientsex': np.random.choice(['M', 'F'], n_samples),
        'reaction': np.random.choice(['nausea', 'headache', 'rash', 'fever', 'dizziness'], n_samples),
        'brand_name': np.random.choice(['drug_a', 'drug_b', 'drug_c', 'drug_d'], n_samples),
        'serious': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
    }
    
    # Ensure age is reasonable
    data['patientonsetage'] = np.clip(data['patientonsetage'], 0, 120)
    
    df = pd.DataFrame(data)
    print(f"Created mock dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from config.config_manager import config_manager
        config = config_manager.load_config()
        print(f"Configuration loaded successfully")
        print(f"  - MLflow experiment: {config.mlflow.experiment_name}")
        print(f"  - Available tasks: {list(config.features.keys())}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging system."""
    print("\nTesting logging...")
    
    try:
        from utils.logger import get_logger
        logger = get_logger("test_logger")
        logger.info("Test log message")
        print("Logging system working")
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def test_model_training():
    """Test model training with mock data."""
    print("\nTesting model training...")
    
    try:
        from training.trainer import MLTrainer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Create mock data
        df = create_mock_data()
        
        # Prepare features and target
        features = ['patientonsetage', 'patientsex', 'reaction', 'brand_name']
        X = df[features]
        y = df['serious']
        
        # Create simple model
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['patientonsetage']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['patientsex', 'reaction', 'brand_name'])
        ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Train model
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = (predictions == y).mean()
        
        print(f"Model training successful")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def test_model_evaluation():
    """Test model evaluation."""
    print("\nTesting model evaluation...")
    
    try:
        from evaluation.model_evaluator import ModelEvaluator
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Create mock data
        df = create_mock_data()
        features = ['patientonsetage', 'patientsex', 'reaction', 'brand_name']
        X = df[features]
        y = df['serious']
        
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Test evaluator
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_classification(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name="test_model"
        )
        
        print(f"Model evaluation successful")
        print(f"  - Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"  - F1 Score: {results['metrics']['f1_macro']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("ML PIPELINE SETUP TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Logging Test", test_logging),
        ("Model Training Test", test_model_training),
        ("Model Evaluation Test", test_model_evaluation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "✗ FAIL"
        print(f"{test_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Your ML pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Set up your database connection (WAREHOUSE_DB_URI)")
        print("2. Run: python examples/full_pipeline_example.py")
        print("3. Start MLflow: mlflow server --backend-store-uri sqlite:///mlflow.db")
        print("4. Start model server: python -m src.cli.main serve")
    else:
        print(f"\n{total - passed} tests failed. Please check the errors above.")
        print("Make sure you've installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
