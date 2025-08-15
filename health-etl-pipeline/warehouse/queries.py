"""
SQL query definitions for the data warehouse.

These queries are designed for use by downstream consumers such as
visualization apps, APIs, and analysis scripts. The idea is to keep
SQL centralized so changes in schema don't require changes everywhere.
"""

# --- Basic exploration queries ---

GET_ALL_DRUGS = """
SELECT
    product_ndc,
    generic_name,
    brand_name,
    dosage_form,
    route,
    marketing_start_date,
    marketing_end_date,
    labeler_name
FROM openfda_drugs;
"""

GET_RECENT_DRUGS = """
SELECT
    product_ndc,
    generic_name,
    brand_name,
    marketing_start_date
FROM openfda_drugs
WHERE marketing_start_date >= CURRENT_DATE - INTERVAL '1 year'
ORDER BY marketing_start_date DESC;
"""

# --- Aggregation / trends ---

COUNT_BY_ROUTE = """
SELECT
    route,
    COUNT(*) AS drug_count
FROM openfda_drugs
GROUP BY route
ORDER BY drug_count DESC;
"""

COUNT_BY_DOSAGE_FORM = """
SELECT
    dosage_form,
    COUNT(*) AS drug_count
FROM openfda_drugs
GROUP BY dosage_form
ORDER BY drug_count DESC;
"""

DRUGS_BY_YEAR = """
SELECT
    EXTRACT(YEAR FROM marketing_start_date) AS year,
    COUNT(*) AS drug_count
FROM openfda_drugs
GROUP BY year
ORDER BY year;
"""

# --- Search queries (parameterized in code) ---

SEARCH_BY_GENERIC_NAME = """
SELECT
    *
FROM openfda_drugs
WHERE generic_name ILIKE %(name_pattern)s;
"""

SEARCH_BY_LABELER = """
SELECT
    *
FROM openfda_drugs
WHERE labeler_name ILIKE %(labeler_pattern)s;
"""
