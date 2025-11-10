from flask import Blueprint, jsonify, request
from services.postgres_client import query_df, trends_by_date
import json

data_bp = Blueprint("data", __name__)

@data_bp.route("/trends", methods=["GET"])
def trends():
    table = request.args.get("table", "openfda_events")
    metric_col = request.args.get("metric_col", "seriousness")
    group_by = request.args.get("group_by", "patientsex")
    date_col = request.args.get("date_col", "receivedate")
    limit = int(request.args.get("limit", 500))
    df = trends_by_date(table, date_col=date_col, metric_col=metric_col, group_by=group_by, limit=limit)
    return df.to_dict(orient="records")

@data_bp.route("/table", methods=["GET"])
def table_view():
    table = request.args.get("table", "openfda_events")
    q = request.args.get("q")
    size = int(request.args.get("size", 100))
    sql = f"SELECT * FROM {table}"
    if q:
        sql += f" WHERE {q}"
    sql += f" LIMIT {size}"
    df = query_df(sql)
    return df.to_dict(orient="records")
