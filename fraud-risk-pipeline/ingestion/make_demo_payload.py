import json
from pathlib import Path

from ingestion.db import get_connection

OUT = Path("exports")
OUT.mkdir(exist_ok=True)
OUT_PATH = OUT / "demo_payload.json"

def main():
    con = get_connection()
    try:
        # pick a high-risk sample from the test slice (last 20%)
        df = con.execute("""
            SELECT *
            FROM features.transactions_features
            ORDER BY amount_zscore_24h DESC
            LIMIT 1
        """).fetchdf()
    finally:
        con.close()

    row = df.iloc[0].to_dict()

    # API expects only feature columns (not labels / helper cols)
    drop = {"is_fraud", "ts", "amount_rollmean_24h", "amount_rollstd_24h"}
    payload = {k: float(v) for k, v in row.items() if k not in drop}

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote demo payload to {OUT_PATH}")

if __name__ == "__main__":
    main()
