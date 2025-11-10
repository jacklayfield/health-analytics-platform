from flask import Blueprint, jsonify, request
from services.airflow_client import list_dags, list_dag_runs, trigger_dag, get_dag

etl_bp = Blueprint("etl", __name__)

@etl_bp.route("/dags", methods=["GET"])
def dags():
    return jsonify(list_dags())

@etl_bp.route("/dags/<dag_id>", methods=["GET"])
def dag_info(dag_id):
    return jsonify(get_dag(dag_id))

@etl_bp.route("/dags/<dag_id>/runs", methods=["GET"])
def dag_runs(dag_id):
    return jsonify(list_dag_runs(dag_id))

@etl_bp.route("/dags/<dag_id>/trigger", methods=["POST"])
def dag_trigger(dag_id):
    body = request.json or {}
    res = trigger_dag(dag_id, conf=body.get("conf"))
    return jsonify(res)
