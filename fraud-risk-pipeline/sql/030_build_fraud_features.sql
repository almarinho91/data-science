CREATE OR REPLACE TABLE features.transactions_features AS
WITH base AS (
    SELECT
        *,
        -- convert seconds to timestamp-like scale (dataset has seconds since first tx)
        Time AS ts
    FROM stg.transactions
),

windowed AS (
    SELECT
        *,
        -- velocity features
        COUNT(*) OVER (
            ORDER BY ts
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ) AS tx_count_1h,

        COUNT(*) OVER (
            ORDER BY ts
            RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
        ) AS tx_count_24h,

        -- rolling amount mean (24h)
        AVG(log_amount) OVER (
            ORDER BY ts
            RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
        ) AS amount_rollmean_24h,

        STDDEV(log_amount) OVER (
            ORDER BY ts
            RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
        ) AS amount_rollstd_24h
    FROM base
),

final AS (
    SELECT
        *,

        -- z-score vs recent behavior
        (log_amount - amount_rollmean_24h) / NULLIF(amount_rollstd_24h, 0) AS amount_zscore_24h,

        -- simple risk flag
        CASE WHEN Amount > 1000 THEN 1 ELSE 0 END AS is_high_amount
    FROM windowed
)

SELECT *
FROM final;
