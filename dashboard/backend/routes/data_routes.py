from flask import Blueprint, jsonify, request
from services.postgres_client import query_df, trends_by_date
from services.synthea_client import get_patients_df, get_conditions_df, get_medications_df
import json
import pandas as pd

data_bp = Blueprint("data", __name__)

def convert_response(df):
    """Convert DataFrame to list of dicts with NaN values replaced by None."""
    data = df.to_dict(orient="records")
    # Replace NaN with None for JSON serialization
    for record in data:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    return data

@data_bp.route("/trends", methods=["GET"])
def trends():
    table = request.args.get("table", "openfda_events")
    metric_col = request.args.get("metric_col", "serious")
    group_by = request.args.get("group_by", "patientsex")
    date_col = request.args.get("date_col", "receivedate")
    limit = int(request.args.get("limit", 500))
    df = trends_by_date(table, date_col=date_col, metric_col=metric_col, group_by=group_by, limit=limit)
    return jsonify(convert_response(df))

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
    return jsonify(convert_response(df))

@data_bp.route("/synthea/patients", methods=["GET"])
def synthea_patients():
    df = get_patients_df()
    return jsonify(convert_response(df))

@data_bp.route("/synthea/conditions", methods=["GET"])
def synthea_conditions():
    df = get_conditions_df()
    return jsonify(convert_response(df))

@data_bp.route("/synthea/medications", methods=["GET"])
def synthea_medications():
    df = get_medications_df()
    return jsonify(convert_response(df))

@data_bp.route("/openfda/events", methods=["GET"])
def openfda_events():
    """Get OpenFDA safety events data."""
    size = int(request.args.get("size", 100))
    try:
        df = query_df(f"SELECT * FROM public.openfda_events LIMIT {size}")
        if df.empty:
            print(f"Warning: openfda_events query returned empty result")
        return jsonify(convert_response(df))
    except Exception as e:
        print(f"Error fetching openfda_events: {e}")
        return jsonify({"error": str(e)})

@data_bp.route("/openfda/patients", methods=["GET"])
def openfda_patients():
    """Get unique patients from OpenFDA events with event counts."""
    limit = int(request.args.get("limit", 50))
    try:
        sql = f"""
            SELECT 
                patientonsetage,
                patientsex,
                COUNT(*) as event_count,
                SUM(serious) as serious_count,
                MIN(receivedate) as first_event,
                MAX(receivedate) as last_event
            FROM public.openfda_events
            GROUP BY patientonsetage, patientsex
            ORDER BY event_count DESC
            LIMIT {limit}
        """
        df = query_df(sql)
        if df.empty:
            print(f"Warning: openfda_patients query returned empty result")
        return jsonify(convert_response(df))
    except Exception as e:
        print(f"Error fetching openfda_patients: {e}")
        return jsonify({"error": str(e)})
  