"""
Tests for model evaluation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.evaluation.model_evaluator import ModelEvaluator
from src.config.config_manager import EvaluationConfig, TrainingConfig


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def mock_eval_config(self):
        """Mock evaluation configuration."""
        return EvaluationConfig(
            metrics={
                'classification': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc'],
                'regression': ['mse', 'mae', 'r2']
            }
        )
    
    @pytest.fixture
    def mock_training_config(self):
        """Mock training configuration."""
        return TrainingConfig(
            test_size=0.2,
            validation_size=0.2,
            random_state=42,
            stratify=True,
            cross_validation={'cv_folds': 5, 'scoring': ['accuracy', 'f1_macro']}
        )
    
    @pytest.fixture
    def sample_classification_data(self):
        """Sample classification data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Create target with some relationship to features
        y = ((X['feature1'] > 0) & (X['feature2'] > 0)).astype(int)
        y = pd.Series(y)
        
        return X, y
    
    @pytest.fixture
    def sample_regression_data(self):
        """Sample regression data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Create target with some relationship to features
        y = 2 * X['feature1'] + 3 * X['feature2'] + np.random.normal(0, 0.1, n_samples)
        y = pd.Series(y)
        
        return X, y
    
    @pytest.fixture
    def mock_classification_model(self, sample_classification_data):
        """Mock classification model."""
        X, y = sample_classification_data
        
        # Create a simple pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['feature1', 'feature2']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['feature3'])
        ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        model.fit(X, y)
        return model
    
    def test_init(self, mock_eval_config, mock_training_config):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        assert evaluator.eval_config == mock_eval_config
        assert evaluator.training_config == mock_training_config
    
    def test_evaluate_classification(self, mock_eval_config, mock_training_config, 
                                   sample_classification_data, mock_classification_model):
        """Test classification model evaluation."""
        X, y = sample_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        results = evaluator.evaluate_classification(
            model=mock_classification_model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            model_name="test_model"
        )
        
        # Check basic structure
        assert 'model_name' in results
        assert 'task_type' in results
        assert 'metrics' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        
        # Check model name
        assert results['model_name'] == "test_model"
        assert results['task_type'] == "classification"
        
        # Check metrics
        assert 'accuracy' in results['metrics']
        assert 'precision_macro' in results['metrics']
        assert 'recall_macro' in results['metrics']
        assert 'f1_macro' in results['metrics']
        
        # Check metric values are reasonable
        assert 0 <= results['metrics']['accuracy'] <= 1
        assert 0 <= results['metrics']['precision_macro'] <= 1
        assert 0 <= results['metrics']['recall_macro'] <= 1
        assert 0 <= results['metrics']['f1_macro'] <= 1
        
        # Check cross-validation results
        assert 'cross_validation' in results
        assert 'cv_accuracy_mean' in results['cross_validation']
        assert 'cv_f1_macro_mean' in results['cross_validation']
    
    def test_evaluate_classification_without_cv(self, mock_eval_config, mock_training_config,
                                              sample_classification_data, mock_classification_model):
        """Test classification evaluation without cross-validation."""
        X, y = sample_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        results = evaluator.evaluate_classification(
            model=mock_classification_model,
            X_test=X_test,
            y_test=y_test,
            model_name="test_model"
        )
        
        # Should not have cross-validation results
        assert 'cross_validation' not in results
        assert 'metrics' in results
        assert 'accuracy' in results['metrics']
    
    def test_evaluate_regression(self, mock_eval_config, mock_training_config,
                               sample_regression_data):
        """Test regression model evaluation."""
        from sklearn.linear_model import LinearRegression
        
        X, y = sample_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create regression model
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['feature1', 'feature2']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['feature3'])
        ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        results = evaluator.evaluate_regression(
            model=model,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            model_name="test_regression_model"
        )
        
        # Check basic structure
        assert 'model_name' in results
        assert 'task_type' in results
        assert 'metrics' in results
        assert 'residuals' in results
        
        # Check model name and type
        assert results['model_name'] == "test_regression_model"
        assert results['task_type'] == "regression"
        
        # Check metrics
        assert 'mse' in results['metrics']
        assert 'rmse' in results['metrics']
        assert 'mae' in results['metrics']
        assert 'r2' in results['metrics']
        assert 'mape' in results['metrics']
        
        # Check metric values are reasonable
        assert results['metrics']['mse'] >= 0
        assert results['metrics']['rmse'] >= 0
        assert results['metrics']['mae'] >= 0
        assert results['metrics']['r2'] <= 1  # R² can be negative but typically ≤ 1
    
    def test_compare_models(self, mock_eval_config, mock_training_config):
        """Test model comparison functionality."""
        # Create mock evaluation results
        model_results = [
            {
                'model_name': 'model1',
                'task_type': 'classification',
                'test_size': 200,
                'accuracy': 0.85,
                'f1_macro': 0.82,
                'precision_macro': 0.83
            },
            {
                'model_name': 'model2',
                'task_type': 'classification',
                'test_size': 200,
                'accuracy': 0.90,
                'f1_macro': 0.88,
                'precision_macro': 0.89
            },
            {
                'model_name': 'model3',
                'task_type': 'classification',
                'test_size': 200,
                'accuracy': 0.87,
                'f1_macro': 0.85,
                'precision_macro': 0.86
            }
        ]
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        comparison_df = evaluator.compare_models(model_results, primary_metric='accuracy')
        
        # Check DataFrame structure
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert 'model_name' in comparison_df.columns
        assert 'accuracy' in comparison_df.columns
        
        # Check sorting (best model first)
        assert comparison_df.iloc[0]['model_name'] == 'model2'  # Highest accuracy
        assert comparison_df.iloc[0]['accuracy'] == 0.90
    
    def test_generate_evaluation_report(self, mock_eval_config, mock_training_config, tmp_path):
        """Test evaluation report generation."""
        evaluation_results = {
            'model_name': 'test_model',
            'task_type': 'classification',
            'metrics': {'accuracy': 0.85, 'f1_macro': 0.82},
            'classification_report': {'0': {'precision': 0.8, 'recall': 0.9}},
            'confusion_matrix': [[80, 20], [10, 90]],
            'test_size': 200
        }
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        with patch('src.evaluation.model_evaluator.Path') as mock_path:
            mock_path.return_value = tmp_path
            report_path = evaluator.generate_evaluation_report(
                evaluation_results, str(tmp_path)
            )
            
            # Check that report was generated
            assert report_path is not None
            assert str(tmp_path) in report_path
    
    def test_plot_evaluation_metrics(self, mock_eval_config, mock_training_config, tmp_path):
        """Test evaluation plot generation."""
        evaluation_results = {
            'model_name': 'test_model',
            'task_type': 'classification',
            'metrics': {
                'accuracy': 0.85,
                'precision_macro': 0.83,
                'recall_macro': 0.87,
                'f1_macro': 0.82
            },
            'confusion_matrix': [[80, 20], [10, 90]]
        }
        
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        with patch('src.evaluation.model_evaluator.Path') as mock_path:
            mock_path.return_value = tmp_path
            plot_paths = evaluator.plot_evaluation_metrics(
                evaluation_results, str(tmp_path)
            )
            
            # Should generate at least confusion matrix plot
            assert len(plot_paths) >= 1
            assert all(str(tmp_path) in path for path in plot_paths)
    
    def test_calculate_classification_metrics_with_probabilities(self, mock_eval_config, mock_training_config):
        """Test classification metrics calculation with probabilities."""
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        
        metrics = evaluator._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'roc_auc' in metrics  # Should be present for binary classification
        
        # Check ROC AUC is reasonable
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_calculate_regression_metrics(self, mock_eval_config, mock_training_config):
        """Test regression metrics calculation."""
        evaluator = ModelEvaluator(mock_eval_config, mock_training_config)
        
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator._calculate_regression_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check that RMSE is square root of MSE
        assert abs(metrics['rmse'] - np.sqrt(metrics['mse'])) < 1e-10
        
        # Check that all metrics are non-negative (except R² which can be negative)
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0

