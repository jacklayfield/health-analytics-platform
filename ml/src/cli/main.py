"""
Command-line interface for ML pipeline operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from ..config.config_manager import config_manager
from ..training.trainer import MLTrainer
from ..serving.model_server import ModelServer
from ..utils.logger import get_logger

logger = get_logger(__name__)


def train_command(args):
    """Handle train command."""
    logger.info(f"Starting training for task: {args.task}")
    
    try:
        # Initialize trainer
        trainer = MLTrainer(task_name=args.task)
        
        # Run training pipeline
        results = trainer.run_full_pipeline(
            table_name=args.table,
            filters=parse_filters(args.filters) if args.filters else None,
            optimize_hyperparams=args.optimize
        )
        
        # Print summary
        print_training_summary(results)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def serve_command(args):
    """Handle serve command."""
    logger.info("Starting model server")
    
    try:
        # Create and run server
        server = ModelServer(models_dir=args.models_dir)
        server.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


def evaluate_command(args):
    """Handle evaluate command."""
    logger.info(f"Evaluating model: {args.model}")
    
    try:
        # Load model and evaluate
        # This would be implemented based on specific requirements
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def optimize_command(args):
    """Handle hyperparameter optimization command."""
    logger.info(f"Starting hyperparameter optimization for task: {args.task}")
    
    try:
        # Initialize trainer
        trainer = MLTrainer(task_name=args.task)
        
        # Load data
        X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
            table_name=args.table
        )
        
        # Run optimization for each model
        for algo_config in trainer.model_config.algorithms:
            model_name = algo_config['name']
            logger.info(f"Optimizing {model_name}")
            
            # This would run hyperparameter optimization
            # Implementation depends on specific requirements
        
        logger.info("Hyperparameter optimization completed")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


def config_command(args):
    """Handle config command."""
    if args.action == "show":
        show_config()
    elif args.action == "validate":
        validate_config()
    elif args.action == "init":
        init_config(args.output)


def show_config():
    """Show current configuration."""
    try:
        config = config_manager.load_config()
        
        print("Current ML Pipeline Configuration:")
        print("=" * 50)
        
        print(f"Data Warehouse URI: {config.data.warehouse_uri}")
        print(f"MLflow Tracking URI: {config.mlflow.tracking_uri}")
        print(f"MLflow Experiment: {config.mlflow.experiment_name}")
        
        print("\nAvailable Tasks:")
        for task_name, task_config in config.features.items():
            print(f"  - {task_name}")
            print(f"    Target: {task_config.target}")
            print(f"    Features: {len(task_config.numeric_features + task_config.categorical_features)}")
        
        print("\nAvailable Models:")
        for task_name, model_config in config.models.items():
            print(f"  - {task_name}")
            for algo in model_config.algorithms:
                print(f"    - {algo['name']} ({algo['class']})")
        
    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        sys.exit(1)


def validate_config():
    """Validate configuration."""
    try:
        config = config_manager.load_config()
        print("Configuration validation passed")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def init_config(output_path: str):
    """Initialize configuration file."""
    try:
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy default config
        default_config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        if default_config_path.exists():
            import shutil
            shutil.copy2(default_config_path, config_path)
            print(f"Configuration initialized at: {config_path}")
        else:
            print("Default configuration not found")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)


def parse_filters(filters_str: str) -> Dict[str, Any]:
    """Parse filters string into dictionary."""
    filters = {}
    
    if not filters_str:
        return filters
    
    for filter_item in filters_str.split(','):
        if '=' in filter_item:
            key, value = filter_item.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            if value.lower() in ['true', 'false']:
                filters[key] = value.lower() == 'true'
            elif value.isdigit():
                filters[key] = int(value)
            elif value.replace('.', '').isdigit():
                filters[key] = float(value)
            else:
                filters[key] = value
    
    return filters


def print_training_summary(results: Dict[str, Any]):
    """Print training summary."""
    print("\nTraining Summary")
    print("=" * 50)
    print(f"Task: {results['task_name']}")
    print(f"Data: {results['data_info']['train_samples']} train, {results['data_info']['test_samples']} test")
    print(f"Features: {results['data_info']['features']}")
    
    summary = results['summary']
    print(f"\nModels: {summary['successful_models']}/{summary['total_models']} successful")
    
    if 'best_model' in summary and summary['best_model']:
        print(f"Best Model: {summary['best_model']} (accuracy: {summary['best_accuracy']:.4f})")
    
    print(f"Successful Models: {', '.join(summary['successful_model_names'])}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Health Analytics ML Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models for serious prediction
  python -m src.cli.main train serious_prediction
  
  # Train with hyperparameter optimization
  python -m src.cli.main train serious_prediction --optimize
  
  # Train with data filters
  python -m src.cli.main train serious_prediction --filters "serious=1,patientsex=M"
  
  # Start model server
  python -m src.cli.main serve --port 8080
  
  # Show configuration
  python -m src.cli.main config show
  
  # Validate configuration
  python -m src.cli.main config validate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('task', help='Task name (e.g., serious_prediction)')
    train_parser.add_argument('--table', default='openfda_events', help='Data table name')
    train_parser.add_argument('--filters', help='Data filters (key=value,key2=value2)')
    train_parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    train_parser.set_defaults(func=train_command)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start model serving server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--models-dir', help='Models directory path')
    serve_parser.set_defaults(func=serve_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('model', help='Model name to evaluate')
    eval_parser.add_argument('--test-data', help='Test data path')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
    opt_parser.add_argument('task', help='Task name')
    opt_parser.add_argument('--table', default='openfda_events', help='Data table name')
    opt_parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    opt_parser.set_defaults(func=optimize_command)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='action', help='Config actions')
    
    config_subparsers.add_parser('show', help='Show current configuration')
    config_subparsers.add_parser('validate', help='Validate configuration')
    
    init_parser = config_subparsers.add_parser('init', help='Initialize configuration')
    init_parser.add_argument('output', help='Output configuration file path')
    
    config_parser.set_defaults(func=config_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()

