import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

SYNTHREA_DATA_PATH = Path("/app/synthea_data")

def load_fhir_bundle(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_patient_data() -> List[Dict[str, Any]]:
    patients = []
    fhir_dir = SYNTHREA_DATA_PATH / "fhir"

    if not fhir_dir.exists():
        return patients

    for file_path in fhir_dir.glob("*.json"):
        if "hospitalInformation" in file_path.name or "practitionerInformation" in file_path.name:
            continue

        bundle = load_fhir_bundle(file_path)
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                name_data = resource.get("name", [{}])[0]
                given_names = name_data.get("given", [])
                family_name = name_data.get("family", "")
                
                patient = {
                    "id": resource.get("id"),
                    "first": " ".join(given_names) if given_names else "",
                    "last": family_name,
                    "gender": resource.get("gender"),
                    "birthdate": resource.get("birthDate"),
                    "address": resource.get("address", [{}])[0].get("city", "") + ", " + resource.get("address", [{}])[0].get("state", ""),
                }
                patients.append(patient)

    return patients

def get_conditions_data() -> List[Dict[str, Any]]:
    conditions = []
    fhir_dir = SYNTHREA_DATA_PATH / "fhir"

    if not fhir_dir.exists():
        return conditions

    for file_path in fhir_dir.glob("*.json"):
        if "hospitalInformation" in file_path.name or "practitionerInformation" in file_path.name:
            continue

        bundle = load_fhir_bundle(file_path)
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Condition":
                condition = {
                    "id": resource.get("id"),
                    "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                    "code": resource.get("code", {}).get("coding", [{}])[0].get("code"),
                    "display": resource.get("code", {}).get("coding", [{}])[0].get("display"),
                    "onsetDateTime": resource.get("onsetDateTime"),
                    "clinicalStatus": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
                }
                conditions.append(condition)

    return conditions

def get_medications_data() -> List[Dict[str, Any]]:
    medications = []
    fhir_dir = SYNTHREA_DATA_PATH / "fhir"

    if not fhir_dir.exists():
        return medications

    for file_path in fhir_dir.glob("*.json"):
        if "hospitalInformation" in file_path.name or "practitionerInformation" in file_path.name:
            continue

        bundle = load_fhir_bundle(file_path)
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "MedicationRequest":
                medication = {
                    "id": resource.get("id"),
                    "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
                    "medication_code": resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code"),
                    "medication_display": resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display"),
                    "authoredOn": resource.get("authoredOn"),
                    "status": resource.get("status"),
                }
                medications.append(medication)

    return medications

def get_patients_df() -> pd.DataFrame:
    """Get patients data as DataFrame."""
    return pd.DataFrame(get_patient_data())

def get_conditions_df() -> pd.DataFrame:
    """Get conditions data as DataFrame."""
    return pd.DataFrame(get_conditions_data())

def get_medications_df() -> pd.DataFrame:
    """Get medications data as DataFrame."""
    return pd.DataFrame(get_medications_data())