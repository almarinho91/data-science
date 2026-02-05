from ingestion.db import get_connection

def main():
    con = get_connection()
    try:
        print(con.execute("""
            SELECT COUNT(*) AS rows,
                   SUM(Class) AS fraud_rows,
                   ROUND(SUM(Class) * 100.0 / COUNT(*), 4) AS fraud_pct
            FROM raw.transactions
        """).fetchdf())

        print(con.execute("""
            SELECT MIN(Amount) AS min_amount,
                   MAX(Amount) AS max_amount,
                   AVG(Amount) AS avg_amount
            FROM raw.transactions
        """).fetchdf())
    finally:
        con.close()

if __name__ == "__main__":
    main()
