from pathlib import Path

import pandas as pd

from ingestion.db import get_connection

DATA_FILE = Path("data/creditcard.csv")

def main():
    if not DATA_FILE.exists():
        raise SystemExit("Missing data/creditcard.csv — run python -m ingestion.download_data first")

    df = pd.read_csv(DATA_FILE)

    con = get_connection()
    try:
        # stage dataframe
        con.execute("CREATE TEMP TABLE tmp AS SELECT * FROM df")

        # add lineage
        con.execute("ALTER TABLE tmp ADD COLUMN source_file STRING")
        con.execute("UPDATE tmp SET source_file = ?", [DATA_FILE.name])

        con.execute("ALTER TABLE tmp ADD COLUMN ingested_at TIMESTAMP")
        con.execute("UPDATE tmp SET ingested_at = current_timestamp")

        # idempotent load: delete by source_file then insert
        con.execute("DELETE FROM raw.transactions WHERE source_file = ?", [DATA_FILE.name])

        con.execute("""
            INSERT INTO raw.transactions
            SELECT * FROM tmp
        """)

        con.execute("DROP TABLE tmp")
    finally:
        con.close()

    print("Loaded into DuckDB: raw.transactions")

if __name__ == "__main__":
    main()
