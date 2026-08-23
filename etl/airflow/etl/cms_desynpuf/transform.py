from pathlib import Path

import pandas as pd


DATE_COLUMNS = {
    "BENE_BIRTH_DT",
    "BENE_DEATH_DT",
    "CLM_FROM_DT",
    "CLM_THRU_DT",
    "CLM_ADMSN_DT",
    "NCH_BENE_DSCHRG_DT",
}


def _read_csv(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        input_path,
        dtype=str,
        na_values=["", "NA", "N/A"],
        keep_default_na=True,
    )
    frame.columns = [column.strip() for column in frame.columns]

    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].str.strip()

    return frame


def _convert_types(frame: pd.DataFrame) -> pd.DataFrame:
    for column in DATE_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_datetime(
            frame[column],
            format="%Y%m%d",
            errors="coerce",
        )

    numeric_columns = [
        column
        for column in frame.columns
        if column.endswith("_AMT")
        or column.endswith("_AM")
        or column.endswith("_CNT")
        or column.endswith("_MNS")
        or column.endswith("_MOS")
        or column in {"CLM_PMT_AMT", "CLM_UTLZTN_DAY_CNT"}
        or column.startswith(("MEDREIMB_", "BENRES_", "PPPYMT_"))
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _transform_beneficiary(frame: pd.DataFrame, source_year: int) -> pd.DataFrame:
    if "BENE_BIRTH_DT" in frame:
        frame["birth_year"] = frame["BENE_BIRTH_DT"].dt.year

    if "BENE_BIRTH_DT" in frame:
        frame["age_at_end_of_source_year"] = (
            source_year - frame["BENE_BIRTH_DT"].dt.year
        )

    cost_columns = [
        column
        for column in (
            "MEDREIMB_IP",
            "BENRES_IP",
            "PPPYMT_IP",
            "MEDREIMB_OP",
            "BENRES_OP",
            "PPPYMT_OP",
            "MEDREIMB_CAR",
            "BENRES_CAR",
            "PPPYMT_CAR",
        )
        if column in frame
    ]
    if cost_columns:
        frame["total_beneficiary_cost"] = frame[cost_columns].sum(axis=1, min_count=1)

    condition_columns = [
        column
        for column in frame.columns
        if column.startswith("SP_") and column not in {"SP_STATE_CODE"}
    ]
    for column in condition_columns:
        frame[f"has_{column[3:].lower()}"] = frame[column].eq("1")

    return frame


def transform_cms_desynpuf(
    input_root: str = "/opt/airflow/data/raw/cms_desynpuf",
    output_root: str = "/opt/airflow/data/processed/cms_desynpuf",
    **kwargs,
) -> dict[str, str]:
    """Transform extracted CMS DE-SynPUF CSV files into typed CSV outputs."""
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_patterns = {
        "beneficiary": "beneficiary_*/*.csv",
        "inpatient": "inpatient/*.csv",
        "outpatient": "outpatient/*.csv",
    }
    outputs = {}

    for dataset_name, pattern in dataset_patterns.items():
        input_files = sorted(input_dir.glob(pattern))
        if not input_files:
            continue

        frames = []
        for input_path in input_files:
            frame = _convert_types(_read_csv(input_path))
            frame["source_file"] = input_path.name
            source_year = None
            if dataset_name == "beneficiary":
                source_year = int(input_path.parent.name.removeprefix("beneficiary_"))
                frame["source_year"] = source_year
                frame = _transform_beneficiary(frame, source_year)
            frames.append(frame)

        output_path = output_dir / f"{dataset_name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
        outputs[dataset_name] = str(output_path)
        print(f"Transformed {dataset_name} data to {output_path}")

    if not outputs:
        raise FileNotFoundError(f"No CMS CSV files found under {input_dir}")

    return outputs
