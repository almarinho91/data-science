from ingestion.db import get_connection

def main():
    con = get_connection()
    con.execute(open("sql/001_create_raw.sql", "r", encoding="utf-8").read())
    con.close()
    print("DuckDB initialized (raw schema + tables).")

if __name__ == "__main__":
    main()
