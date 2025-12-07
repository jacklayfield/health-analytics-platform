#!/usr/bin/env python3
"""
Environment setup script for the ML pipeline.
This script helps you set up the minimal environment needed to run the pipeline.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("  Please use Python 3.9 or higher")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = [
        "models",
        "logs", 
        "artifacts",
        "data",
        "expectations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment_variables():
    """Set up environment variables."""
    print("\nSetting up environment variables...")
    
    # Default environment variables
    env_vars = {
        "WAREHOUSE_DB_URI": "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow",
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
    }
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("# ML Pipeline Environment Variables\n")
            f.write("# Update these values according to your setup\n\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"Created .env file with default values")
        print("  Please update the values in .env according to your setup")
    else:
        print(".env file already exists")

def test_imports():
    """Test that key modules can be imported."""
    print("\nTesting imports...")
    
    try:
        # Add src to path
        sys.path.append(str(Path("src")))
        
        from config.config_manager import config_manager
        from utils.logger import get_logger
        from training.trainer import MLTrainer
        
        print("Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration if it doesn't exist."""
    print("\nChecking configuration...")
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("Configuration file not found")
        print("  Run: python -m src.cli.main config init config/config.yaml")
    else:
        print("Configuration file exists")

def main():
    """Main setup function."""
    print("="*60)
    print("ML PIPELINE ENVIRONMENT SETUP")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("requirements.txt not found")
        print("  Please run this script from the ml/ directory")
        return False
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Create Directories", create_directories),
        ("Setup Environment Variables", setup_environment_variables),
        ("Test Imports", test_imports),
        ("Check Configuration", create_sample_config),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        print("-" * 40)
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            print(f"{step_name} failed: {e}")
    
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if success_count == len(steps):
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update .env file with your database connection")
        print("2. Run: python test_setup.py")
        print("3. Run: python examples/full_pipeline_example.py")
        print("4. Start MLflow: mlflow server --backend-store-uri sqlite:///mlflow.db")
        print("5. Start model server: python -m src.cli.main serve")
    else:
        print(f"Setup completed with {len(steps) - success_count} issues")
        print("Please check the errors above and fix them before proceeding")
    
    return success_count == len(steps)

if __name__ == "__main__":
    main()
