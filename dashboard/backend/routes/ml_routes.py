from flask import Blueprint, jsonify, request
from services.mlflow_client import list_experiments, latest_run_for_experiment, get_run_metrics, list_registered_models
from config import ML_CONFIG

ml_bp = Blueprint("ml", __name__)

@ml_bp.route("/experiments", methods=["GET"])
def experiments():
    return jsonify(list_experiments())

@ml_bp.route("/experiments/latest", methods=["GET"])
def latest_experiment():
    name = request.args.get("name", ML_CONFIG.get("ml", {}).get("experiment_name"))
    run = latest_run_for_experiment(name)
    if not run:
        return jsonify({"error": "no runs"}), 404
    return jsonify(run)

@ml_bp.route("/runs/<run_id>/metrics", methods=["GET"])
def run_metrics(run_id):
    metrics = get_run_metrics(run_id)
    return jsonify(metrics)

@ml_bp.route("/models", methods=["GET"])
def models():
    return jsonify(list_registered_models())
