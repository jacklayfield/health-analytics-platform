SELECT TO_CHAR(TO_DATE(receivedate, 'YYYYMMDD'), 'YYYY-MM') AS month,
       COUNT(*) AS reports
FROM openfda_events
GROUP BY month
ORDER BY month;