import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 checksum for a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def safe_extract_zip(zip_file: zipfile.ZipFile, destination: Path) -> None:
    """Safely extract a ZIP archive."""
    destination = destination.resolve()

    for member in zip_file.namelist():
        member_path = (destination / member).resolve()

        if not str(member_path).startswith(str(destination)):
            raise ValueError(f"Unsafe path detected: {member}")

    zip_file.extractall(destination)


def download_file(url: str, destination: Path) -> None:
    """Download a file to the specified destination."""
    with requests.get(
        url,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> list[dict]:
    """Extract a ZIP archive and inventory its files."""

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Invalid ZIP archive: {archive_path}")

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        safe_extract_zip(zip_ref, destination)

    files = []

    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(destination)

        files.append(
            {
                "name": str(relative_path),
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )

    return files


def extract_cms_desynpuf(
    config_path: str = "/opt/airflow/etl/cms_desynpuf/config/cms_desynpuf.yaml",
    raw_root: str = "/opt/airflow/data/raw/cms_desynpuf",
    **kwargs,
):
    """Download and extract CMS DE-SynPUF files defined in YAML config."""

    config_file = Path(config_path)
    raw_dir = Path(raw_root)

    raw_dir.mkdir(parents=True, exist_ok=True)

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = config["dataset"]
    sample = config["sample"]

    run_timestamp = datetime.now(timezone.utc).isoformat()

    manifest_files = []

    # Process beneficiary files.
    for item in config.get("beneficiary", []):
        year = item["year"]
        url = item["url"]

        dataset_name = f"beneficiary_{year}"
        archive_path = raw_dir / f"{dataset_name}.zip"
        extracted_dir = raw_dir / dataset_name

        print(f"Processing {dataset_name}...")

        download_file(url, archive_path)

        archive_sha256 = calculate_sha256(archive_path)

        extracted_files = extract_archive(
            archive_path,
            extracted_dir,
        )

        manifest_files.append(
            {
                "type": "beneficiary",
                "year": year,
                "name": dataset_name,
                "source_url": url,
                "archive": archive_path.name,
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "extracted_to": str(extracted_dir),
                "file_count": len(extracted_files),
                "files": extracted_files,
            }
        )

    # Process claim and event files.
    for dataset_type in [
        "inpatient",
        "outpatient",
        "pde",
    ]:
        item = config.get(dataset_type)

        if not item:
            continue

        url = item["url"]

        archive_path = raw_dir / f"{dataset_type}.zip"
        extracted_dir = raw_dir / dataset_type

        print(f"Processing {dataset_type}...")

        download_file(url, archive_path)

        archive_sha256 = calculate_sha256(archive_path)

        extracted_files = extract_archive(
            archive_path,
            extracted_dir,
        )

        manifest_files.append(
            {
                "type": dataset_type,
                "name": dataset_type,
                "source_url": url,
                "archive": archive_path.name,
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "extracted_to": str(extracted_dir),
                "file_count": len(extracted_files),
                "files": extracted_files,
            }
        )

    # Process carrier files.
    for item in config.get("carrier", []):
        part = item["part"]
        url = item["url"]

        dataset_name = f"carrier_{part}"
        archive_path = raw_dir / f"{dataset_name}.zip"
        extracted_dir = raw_dir / dataset_name

        print(f"Processing {dataset_name}...")

        download_file(url, archive_path)

        archive_sha256 = calculate_sha256(archive_path)

        extracted_files = extract_archive(
            archive_path,
            extracted_dir,
        )

        manifest_files.append(
            {
                "type": "carrier",
                "part": part,
                "name": dataset_name,
                "source_url": url,
                "archive": archive_path.name,
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "extracted_to": str(extracted_dir),
                "file_count": len(extracted_files),
                "files": extracted_files,
            }
        )

    # Write manifest.
    manifest = {
        "dataset": dataset,
        "sample": sample,
        "status": "extracted",
        "run_timestamp": run_timestamp,
        "files": manifest_files,
    }

    manifest_path = raw_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"CMS DE-SynPUF Sample {sample} extraction complete.")

    return manifest
