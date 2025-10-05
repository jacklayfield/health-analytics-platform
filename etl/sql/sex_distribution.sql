SELECT 
    CASE patientsex
        WHEN '1' THEN 'Male'
        WHEN '2' THEN 'Female'
        ELSE 'Unknown'
    END AS sex,
    COUNT(*) AS count
FROM openfda_events
GROUP BY sex;