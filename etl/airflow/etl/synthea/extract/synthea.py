import subprocess
from pathlib import Path

def extract_synthea(**kwargs):
    """Run Synthea to generate FHIR data."""
    script_dir = Path(__file__).resolve().parent.parent
    run_synthea_path = script_dir / "run_synthea.py"
    
    result = subprocess.run([
        "python", str(run_synthea_path),
        "-p", "100" 
    ], capture_output=True, text=True, cwd=script_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"Synthea failed: {result.stderr}")
    
    print("Synthea extraction completed")
    print(result.stdout)