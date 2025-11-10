import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, List
from config import MLFLOW_TRACKING_URI, ML_CONFIG, MODEL_REGISTRY

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_client = MlflowClient()

def list_experiments() -> List[Dict[str, Any]]:
    exps = _client.list_experiments()
    return [{"id": e.experiment_id, "name": e.name, "artifact_location": e.artifact_location} for e in exps]

def latest_run_for_experiment(experiment_name: str):
    exps = [e for e in _client.list_experiments() if e.name == experiment_name]
    if not exps:
        return None
    exp_id = exps[0].experiment_id
    runs = _client.search_runs(experiment_ids=[exp_id], order_by=["attributes.start_time DESC"], max_results=1)
    return _run_to_dict(runs[0]) if runs else None

def get_run_metrics(run_id: str):
    run = _client.get_run(run_id)
    return run.data.metrics

def list_registered_models():
    models = _client.search_model_versions(f"name='{MODEL_REGISTRY}'")
    # grouping by version
    out = []
    for mv in models:
        out.append({
            "name": mv.name,
            "version": mv.version,
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "status": getattr(mv, "status", None),
        })
    return out

def _run_to_dict(run):
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }