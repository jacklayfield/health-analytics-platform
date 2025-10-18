#!/usr/bin/env python3
"""
Local CI/CD script that runs the same checks as GitHub Actions.
Use this if you want to avoid GitHub Actions costs.
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run local CI/CD pipeline."""
    print("="*60)
    print("LOCAL CI/CD PIPELINE")
    print("="*60)
    
    # Change to ml directory
    os.chdir(Path(__file__).parent.parent)
    
    steps = [
        ("python -m pip install --upgrade pip", "Upgrade pip"),
        ("pip install -r requirements.txt", "Install dependencies"),
        ("python test_setup.py", "Run tests"),
        ("flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics", "Lint code"),
        ("python -m src.cli.main config validate", "Validate configuration"),
        ("python examples/full_pipeline_example.py", "Run full pipeline test"),
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"\n❌ CI/CD failed at: {description}")
            break
    
    print("\n" + "="*60)
    print("CI/CD SUMMARY")
    print("="*60)
    
    if success_count == len(steps):
        print("🎉 All CI/CD checks passed!")
        print("Your code is ready for deployment.")
    else:
        print(f"⚠️ CI/CD failed: {success_count}/{len(steps)} steps passed")
        print("Please fix the issues before deploying.")
    
    return success_count == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

