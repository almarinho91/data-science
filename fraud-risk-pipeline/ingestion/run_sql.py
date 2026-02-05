from pathlib import Path
from ingestion.db import get_connection

def main():
    con = get_connection()
    try:
        for f in sorted(Path("sql").glob("*.sql")):
            print(f"Running {f.name}")
            con.execute(f.read_text(encoding="utf-8"))
    finally:
        con.close()

    print("SQL transformations completed.")

if __name__ == "__main__":
    main()
