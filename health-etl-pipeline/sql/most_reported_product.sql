SELECT LOWER(medicinalproduct) AS drug_name, COUNT(*) AS count
FROM openfda_events
GROUP BY drug_name
ORDER BY count DESC
LIMIT 10;