from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "songs.db"


def create_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                Track TEXT PRIMARY KEY,
                bpm REAL,
                dynamic_variation_score REAL,
                sub_bass REAL,
                bass REAL,
                low_mid REAL,
                mid REAL,
                high_mid REAL,
                presence REAL,
                brilliance REAL,
                spectral_spread REAL,
                spectral_balance REAL,
                timing_deviation REAL,
                asymmetry_bias REAL,
                asymmetry_index REAL,
                rating REAL
            );
        """)

        conn.commit()

    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    create_database()