SELECT patientonsetage::int / 10 * 10 AS age_group,
       COUNT(*) AS count
FROM openfda_events
WHERE patientonsetage IS NOT NULL
GROUP BY age_group
ORDER BY age_group;