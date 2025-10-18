"""
Model monitoring and drift detection framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.config_manager import config_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """Model monitoring and drift detection framework."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model monitor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.monitoring_data = defaultdict(list)
        self.baseline_stats = {}
        self.drift_thresholds = {
            'ks_test_p_value': 0.05,
            'psi_threshold': 0.2,
            'accuracy_drop_threshold': 0.05,
            'prediction_drift_threshold': 0.1
        }
        
        logger.info(f"Model monitor initialized with models directory: {self.models_dir}")
    
    def set_baseline(
        self,
        model_name: str,
        X_baseline: pd.DataFrame,
        y_baseline: pd.Series,
        predictions_baseline: Optional[np.ndarray] = None
    ) -> None:
        """
        Set baseline statistics for drift detection.
        
        Args:
            model_name: Name of the model
            X_baseline: Baseline feature data
            y_baseline: Baseline target data
            predictions_baseline: Baseline predictions (optional)
        """
        logger.info(f"Setting baseline for model: {model_name}")
        
        # Calculate baseline statistics
        baseline_stats = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X_baseline.shape,
            'feature_stats': self._calculate_feature_statistics(X_baseline),
            'target_distribution': y_baseline.value_counts().to_dict(),
            'predictions_stats': None
        }
        
        # Add prediction statistics if provided
        if predictions_baseline is not None:
            baseline_stats['predictions_stats'] = self._calculate_prediction_statistics(predictions_baseline)
        
        self.baseline_stats[model_name] = baseline_stats
        
        logger.info(f"Baseline set for {model_name} with {len(X_baseline)} samples")
    
    def _calculate_feature_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics for drift detection."""
        stats_dict = {}
        
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                # Numeric features
                stats_dict[column] = {
                    'mean': float(X[column].mean()),
                    'std': float(X[column].std()),
                    'min': float(X[column].min()),
                    'max': float(X[column].max()),
                    'median': float(X[column].median()),
                    'q25': float(X[column].quantile(0.25)),
                    'q75': float(X[column].quantile(0.75)),
                    'missing_ratio': float(X[column].isnull().sum() / len(X))
                }
            else:
                # Categorical features
                value_counts = X[column].value_counts()
                stats_dict[column] = {
                    'unique_values': int(X[column].nunique()),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_ratio': float(value_counts.iloc[0] / len(X)) if len(value_counts) > 0 else 0.0,
                    'missing_ratio': float(X[column].isnull().sum() / len(X)),
                    'value_distribution': value_counts.head(10).to_dict()
                }
        
        return stats_dict
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction statistics."""
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'prediction_distribution': np.bincount(predictions.astype(int)).tolist()
        }
    
    def detect_data_drift(
        self,
        model_name: str,
        X_current: pd.DataFrame,
        drift_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Detect data drift between baseline and current data.
        
        Args:
            model_name: Name of the model
            X_current: Current feature data
            drift_type: Type of drift to detect ("all", "statistical", "distribution")
            
        Returns:
            Drift detection results
        """
        if model_name not in self.baseline_stats:
            raise ValueError(f"No baseline found for model: {model_name}")
        
        logger.info(f"Detecting data drift for model: {model_name}")
        
        baseline_stats = self.baseline_stats[model_name]
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'overall_drift': False
        }
        
        # Detect drift for each feature
        for column in X_current.columns:
            if column in baseline_stats['feature_stats']:
                feature_drift = self._detect_feature_drift(
                    baseline_stats['feature_stats'][column],
                    X_current[column],
                    column
                )
                drift_results['feature_drift'][column] = feature_drift
                
                if feature_drift['drift_detected']:
                    drift_results['drift_detected'] = True
        
        # Calculate overall drift score
        drift_scores = [
            result['drift_score'] for result in drift_results['feature_drift'].values()
            if result['drift_detected']
        ]
        
        if drift_scores:
            drift_results['drift_score'] = np.mean(drift_scores)
            drift_results['overall_drift'] = drift_results['drift_score'] > self.drift_thresholds['psi_threshold']
        
        logger.info(f"Data drift detection completed. Drift detected: {drift_results['drift_detected']}")
        
        return drift_results
    
    def _detect_feature_drift(
        self,
        baseline_stats: Dict[str, Any],
        current_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Detect drift for a single feature."""
        drift_result = {
            'feature_name': feature_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': None,
            'details': {}
        }
        
        # Check if feature is numeric or categorical
        if 'mean' in baseline_stats:  # Numeric feature
            drift_result.update(self._detect_numeric_drift(baseline_stats, current_data))
        else:  # Categorical feature
            drift_result.update(self._detect_categorical_drift(baseline_stats, current_data))
        
        return drift_result
    
    def _detect_numeric_drift(
        self,
        baseline_stats: Dict[str, Any],
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """Detect drift for numeric features."""
        drift_result = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': 'statistical',
            'details': {}
        }
        
        # Remove missing values for analysis
        current_clean = current_data.dropna()
        
        if len(current_clean) == 0:
            drift_result['drift_detected'] = True
            drift_result['drift_type'] = 'missing_data'
            drift_result['details']['error'] = 'All values are missing'
            return drift_result
        
        # Statistical tests
        try:
            # Kolmogorov-Smirnov test (if we had baseline data)
            # For now, use mean and std comparison
            current_mean = current_clean.mean()
            current_std = current_clean.std()
            
            baseline_mean = baseline_stats['mean']
            baseline_std = baseline_stats['std']
            
            # Calculate drift metrics
            mean_drift = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
            std_drift = abs(current_std - baseline_std) / baseline_std if baseline_std > 0 else 0
            
            drift_score = max(mean_drift, std_drift)
            drift_result['drift_score'] = drift_score
            
            # Check if drift exceeds threshold
            if drift_score > self.drift_thresholds['prediction_drift_threshold']:
                drift_result['drift_detected'] = True
                drift_result['details'] = {
                    'baseline_mean': baseline_mean,
                    'current_mean': current_mean,
                    'baseline_std': baseline_std,
                    'current_std': current_std,
                    'mean_drift': mean_drift,
                    'std_drift': std_drift
                }
        
        except Exception as e:
            logger.warning(f"Error in numeric drift detection: {e}")
            drift_result['drift_detected'] = True
            drift_result['details']['error'] = str(e)
        
        return drift_result
    
    def _detect_categorical_drift(
        self,
        baseline_stats: Dict[str, Any],
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """Detect drift for categorical features."""
        drift_result = {
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': 'distribution',
            'details': {}
        }
        
        # Remove missing values
        current_clean = current_data.dropna()
        
        if len(current_clean) == 0:
            drift_result['drift_detected'] = True
            drift_result['drift_type'] = 'missing_data'
            drift_result['details']['error'] = 'All values are missing'
            return drift_result
        
        # Calculate current distribution
        current_dist = current_clean.value_counts(normalize=True)
        baseline_dist = baseline_stats['value_distribution']
        
        # Calculate PSI (Population Stability Index)
        psi_score = self._calculate_psi(baseline_dist, current_dist.to_dict())
        drift_result['drift_score'] = psi_score
        
        # Check if drift exceeds threshold
        if psi_score > self.drift_thresholds['psi_threshold']:
            drift_result['drift_detected'] = True
            drift_result['details'] = {
                'baseline_distribution': baseline_dist,
                'current_distribution': current_dist.to_dict(),
                'psi_score': psi_score
            }
        
        return drift_result
    
    def _calculate_psi(self, baseline_dist: Dict[str, float], current_dist: Dict[str, float]) -> float:
        """Calculate Population Stability Index (PSI)."""
        psi = 0.0
        
        # Get all unique values
        all_values = set(baseline_dist.keys()) | set(current_dist.keys())
        
        for value in all_values:
            baseline_ratio = baseline_dist.get(value, 0.0001)  # Avoid division by zero
            current_ratio = current_dist.get(value, 0.0001)
            
            if baseline_ratio > 0 and current_ratio > 0:
                psi += (current_ratio - baseline_ratio) * np.log(current_ratio / baseline_ratio)
        
        return psi
    
    def detect_performance_drift(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect performance drift in model predictions.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Performance drift results
        """
        logger.info(f"Detecting performance drift for model: {model_name}")
        
        # Calculate current performance metrics
        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            current_metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Store monitoring data
        self.monitoring_data[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'sample_size': len(y_true)
        })
        
        # Detect performance drift
        performance_drift = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'current_metrics': current_metrics,
            'performance_drift_detected': False,
            'drift_details': {}
        }
        
        # Compare with historical performance (if available)
        if len(self.monitoring_data[model_name]) > 1:
            historical_metrics = self._get_historical_metrics(model_name)
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in historical_metrics:
                    historical_mean = np.mean(historical_metrics[metric_name])
                    performance_drop = historical_mean - current_value
                    
                    if performance_drop > self.drift_thresholds['accuracy_drop_threshold']:
                        performance_drift['performance_drift_detected'] = True
                        performance_drift['drift_details'][metric_name] = {
                            'historical_mean': historical_mean,
                            'current_value': current_value,
                            'performance_drop': performance_drop
                        }
        
        logger.info(f"Performance drift detection completed. Drift detected: {performance_drift['performance_drift_detected']}")
        
        return performance_drift
    
    def _get_historical_metrics(self, model_name: str, window_size: int = 10) -> Dict[str, List[float]]:
        """Get historical metrics for drift comparison."""
        historical_data = self.monitoring_data[model_name][-window_size:-1]  # Exclude current
        
        historical_metrics = defaultdict(list)
        for data_point in historical_data:
            for metric_name, value in data_point['metrics'].items():
                historical_metrics[metric_name].append(value)
        
        return dict(historical_metrics)
    
    def generate_monitoring_report(
        self,
        model_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive monitoring report.
        
        Args:
            model_name: Name of the model
            output_path: Output path for report
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = f"monitoring_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'baseline_stats': self.baseline_stats.get(model_name, {}),
            'monitoring_data': self.monitoring_data.get(model_name, []),
            'drift_thresholds': self.drift_thresholds,
            'summary': self._generate_monitoring_summary(model_name)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to: {output_path}")
        return output_path
    
    def _generate_monitoring_summary(self, model_name: str) -> Dict[str, Any]:
        """Generate monitoring summary."""
        summary = {
            'total_monitoring_points': len(self.monitoring_data.get(model_name, [])),
            'has_baseline': model_name in self.baseline_stats,
            'last_monitoring_time': None,
            'performance_trend': 'stable'
        }
        
        if model_name in self.monitoring_data and self.monitoring_data[model_name]:
            last_point = self.monitoring_data[model_name][-1]
            summary['last_monitoring_time'] = last_point['timestamp']
            
            # Analyze performance trend
            if len(self.monitoring_data[model_name]) >= 3:
                recent_accuracy = [point['metrics']['accuracy'] for point in self.monitoring_data[model_name][-3:]]
                if all(recent_accuracy[i] >= recent_accuracy[i-1] for i in range(1, len(recent_accuracy))):
                    summary['performance_trend'] = 'improving'
                elif all(recent_accuracy[i] <= recent_accuracy[i-1] for i in range(1, len(recent_accuracy))):
                    summary['performance_trend'] = 'declining'
        
        return summary
    
    def plot_drift_analysis(
        self,
        model_name: str,
        drift_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate drift analysis plots.
        
        Args:
            model_name: Name of the model
            drift_results: Results from drift detection
            output_path: Output path for plots
            
        Returns:
            Path to generated plots
        """
        if output_path is None:
            output_path = f"drift_analysis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Create subplots
        n_features = len(drift_results['feature_drift'])
        if n_features == 0:
            logger.warning("No features to plot")
            return output_path
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Drift Analysis - {model_name}', fontsize=16)
        
        # Plot 1: Drift scores by feature
        features = list(drift_results['feature_drift'].keys())
        drift_scores = [drift_results['feature_drift'][f]['drift_score'] for f in features]
        
        axes[0, 0].bar(features, drift_scores)
        axes[0, 0].axhline(y=self.drift_thresholds['psi_threshold'], color='r', linestyle='--', label='Threshold')
        axes[0, 0].set_title('Drift Scores by Feature')
        axes[0, 0].set_ylabel('Drift Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # Plot 2: Drift detection summary
        drift_detected = [drift_results['feature_drift'][f]['drift_detected'] for f in features]
        drift_counts = {'No Drift': drift_detected.count(False), 'Drift Detected': drift_detected.count(True)}
        
        axes[0, 1].pie(drift_counts.values(), labels=drift_counts.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Drift Detection Summary')
        
        # Plot 3: Performance over time (if available)
        if model_name in self.monitoring_data and self.monitoring_data[model_name]:
            timestamps = [datetime.fromisoformat(point['timestamp']) for point in self.monitoring_data[model_name]]
            accuracies = [point['metrics']['accuracy'] for point in self.monitoring_data[model_name]]
            
            axes[1, 0].plot(timestamps, accuracies, marker='o')
            axes[1, 0].set_title('Model Performance Over Time')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No performance data available', ha='center', va='center')
            axes[1, 0].set_title('Model Performance Over Time')
        
        # Plot 4: Overall drift score
        axes[1, 1].bar(['Overall Drift Score'], [drift_results['drift_score']])
        axes[1, 1].axhline(y=self.drift_thresholds['psi_threshold'], color='r', linestyle='--', label='Threshold')
        axes[1, 1].set_title('Overall Drift Score')
        axes[1, 1].set_ylabel('Drift Score')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drift analysis plots saved to: {output_path}")
        return output_path
    
    def set_drift_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update drift detection thresholds."""
        self.drift_thresholds.update(thresholds)
        logger.info(f"Drift thresholds updated: {self.drift_thresholds}")
    
    def get_monitoring_summary(self, model_name: str) -> Dict[str, Any]:
        """Get monitoring summary for a model."""
        return self._generate_monitoring_summary(model_name)

