SELECT reaction, COUNT(*) AS count
FROM openfda_events
GROUP BY reaction
ORDER BY count DESC
LIMIT 10;