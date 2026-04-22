#!/usr/bin/env python3
"""
Movies in the Park - Load
Two public interfaces:
  run_load_csv()      — TEST MODE  — writes DataFrames to CSV
  run_load_postgres() — PRODUCTION — writes to ZipCodes, Parks, Events tables
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR           = os.environ.get("DATA_DIR", "/app/data")
OUTPUT_FILE        = os.path.join(DATA_DIR, "cleaned_movies_final.csv")
CENSUS_OUTPUT_FILE = os.path.join(DATA_DIR, "census_data.csv")

# =============================================================================
# TABLE DEFINITIONS
# =============================================================================
CREATE_ZIPCODES_TABLE = """
    CREATE TABLE IF NOT EXISTS ZipCodes (
        zip_code VARCHAR(10) PRIMARY KEY,
        income   INTEGER
    );
"""

CREATE_PARKS_TABLE = """
    CREATE TABLE IF NOT EXISTS Parks (
        park_id   SERIAL PRIMARY KEY,
        park_name VARCHAR(255) NOT NULL,
        address   VARCHAR(255),
        zip_code  VARCHAR(10) NOT NULL,
        FOREIGN KEY (zip_code) REFERENCES ZipCodes(zip_code)
    );
"""

CREATE_EVENTS_TABLE = """
    CREATE TABLE IF NOT EXISTS Events (
        event_id          SERIAL PRIMARY KEY,
        movie_name        VARCHAR(255) NOT NULL,
        park_id           INTEGER      NOT NULL,
        rating            VARCHAR(10)  NOT NULL,
        date              DATE,
        closed_captioning BOOLEAN,
        FOREIGN KEY (park_id) REFERENCES Parks(park_id)
    );
"""

# =============================================================================
# TABLE CREATION
# =============================================================================
def create_tables(conn):
    logger.info("Creating tables...")
    with conn.cursor() as cur:
        logger.info("  Creating ZipCodes table...")
        cur.execute(CREATE_ZIPCODES_TABLE)
        logger.info("  Creating Parks table...")
        cur.execute(CREATE_PARKS_TABLE)
        logger.info("  Creating Events table...")
        cur.execute(CREATE_EVENTS_TABLE)
    conn.commit()
    logger.info("All tables created successfully")

# =============================================================================
# LOAD — ZIPCODES
# =============================================================================
def ensure_all_movie_zips(conn, movies_df: pd.DataFrame):
    """
    Before loading Parks, check every zip code in the movies DataFrame
    exists in ZipCodes. Insert any missing ones with NULL income.
    
    This handles zip codes that failed the Census API call
    (e.g. 60627, 60635 return empty responses from ACS 2019).
    """
    logger.info("  Ensuring all movie zip codes exist in ZipCodes...")

    # Get zip codes already in ZipCodes table
    with conn.cursor() as cur:
        cur.execute("SELECT zip_code FROM ZipCodes;")
        existing_zips = {row[0] for row in cur.fetchall()}

    # Get all unique zip codes from the movies DataFrame
    movie_zips = set(
        str(int(z)).zfill(5)
        for z in movies_df["Zip"].dropna().unique()
        if z != 0
    )

    # Find any that are missing from ZipCodes
    missing_zips = movie_zips - existing_zips

    if not missing_zips:
        logger.info("  All movie zip codes already present in ZipCodes")
        return

    logger.warning(
        f"  {len(missing_zips)} zip codes in movies not found in ZipCodes "
        f"(likely failed Census API calls): {sorted(missing_zips)}"
    )

    with conn.cursor() as cur:
        for zip_code in sorted(missing_zips):
            cur.execute(
                """
                INSERT INTO ZipCodes (zip_code, income)
                VALUES (%s, NULL)
                ON CONFLICT DO NOTHING;
                """,
                (zip_code,)
            )
    conn.commit()
    logger.info(f"  Inserted {len(missing_zips)} missing zip codes with NULL income")


def ensure_unknown_zip(conn):
    """
    Insert a placeholder zip '00000' for parks with no resolvable zip code.
    Must exist in ZipCodes before Parks can reference it via FK.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ZipCodes (zip_code, income)
            VALUES ('00000', NULL)
            ON CONFLICT DO NOTHING;
        """)
    conn.commit()
    logger.info("  Placeholder zip '00000' ensured in ZipCodes")


def load_zipcodes(conn, census_df: pd.DataFrame):
    """
    Populate ZipCodes from census DataFrame.
    Groups by zip_code, averages median household income.

    Bug fix: B19013_001E comes back as strings from the Census API —
    must cast to numeric before calling .mean()
    """
    logger.info("Loading ZipCodes table...")

    # FIX 1 — cast to numeric before aggregation
    census_df = census_df.copy()
    census_df["B19013_001E"] = pd.to_numeric(census_df["B19013_001E"], errors="coerce")

    unique_zips = census_df.groupby("zip_code", as_index=False).agg(
        avg_median_income=("B19013_001E", "mean")
    )

    rows = []
    for _, row in unique_zips.iterrows():
        zip_code = str(row["zip_code"]).strip()
        income   = (
            int(row["avg_median_income"])
            if pd.notna(row["avg_median_income"])
            else None
        )
        rows.append((zip_code, income))

    with conn.cursor() as cur:
        for zip_code, income in rows:
            # FIX 2 — pass values tuple to execute
            cur.execute(
                """
                INSERT INTO ZipCodes (zip_code, income)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """,
                (zip_code, income)   # ← was missing
            )

    conn.commit()
    logger.info(f"  {len(rows):,} zip codes loaded into ZipCodes")

# =============================================================================
# LOAD — PARKS
# =============================================================================
def load_parks(conn, movies_df: pd.DataFrame) -> dict:
    """
    Populate Parks from unique park names in the clean movies DataFrame.
    Returns a dict mapping park_name → park_id for use when loading Events.
    """
    logger.info("Loading Parks table...")

    unique_parks = (
        movies_df[["Park Name", "Address", "Zip"]]
        .drop_duplicates(subset=["Park Name"])
        .dropna(subset=["Park Name"])
    )

    rows = []
    for _, row in unique_parks.iterrows():
        park_name = str(row["Park Name"]).strip()
        address   = (
            str(row["Address"]).strip()
            if pd.notna(row["Address"])
            else "Address Unknown"
        )
        # FIX 3 — use "00000" placeholder which is guaranteed to exist in ZipCodes
        zip_code  = str(int(row["Zip"])).zfill(5) if (pd.notna(row["Zip"]) and row["Zip"] != 0) else "00000"
        rows.append((park_name, address, zip_code))

    park_name_to_id = {}
    with conn.cursor() as cur:
        for park_name, address, zip_code in rows:
            cur.execute(
                """
                INSERT INTO Parks (park_name, address, zip_code)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING park_id, park_name;
                """,
                (park_name, address, zip_code)
            )
            result = cur.fetchone()
            if result:
                park_name_to_id[result[1]] = result[0]

        # Catch any parks that already existed — ON CONFLICT skips RETURNING
        cur.execute("SELECT park_id, park_name FROM Parks;")
        for park_id, park_name in cur.fetchall():
            park_name_to_id[park_name] = park_id

    conn.commit()
    logger.info(f"  {len(rows):,} parks processed — {len(park_name_to_id):,} in Parks table")
    return park_name_to_id

# =============================================================================
# LOAD — EVENTS
# =============================================================================
def load_events(conn, movies_df: pd.DataFrame, park_name_to_id: dict):
    """
    Populate Events using park_id foreign keys resolved from park_name_to_id.
    Skips rows where park_id cannot be resolved.
    """
    logger.info(f"Loading Events table from {len(movies_df):,} clean rows...")

    rows    = []
    skipped = 0

    for _, row in movies_df.iterrows():
        park_name = str(row["Park Name"]).strip() if pd.notna(row["Park Name"]) else None
        park_id   = park_name_to_id.get(park_name)

        if not park_id:
            logger.warning(f"  Skipping event — no park_id found for '{park_name}'")
            skipped += 1
            continue

        movie_name        = str(row["Movie Name"]).strip()
        rating            = str(row["Rating"]).strip() if pd.notna(row["Rating"]) else "NR"
        date              = row["Date"] if pd.notna(row["Date"]) else None

        cc_raw = row["Closed Captioning"]
        if pd.isna(cc_raw):
            closed_captioning = False
        elif str(cc_raw).strip().upper() in ("Y", "YES", "TRUE", "1"):
            closed_captioning = True
        else:
            closed_captioning = False

        rows.append((movie_name, park_id, rating, date, closed_captioning))

    insert_sql = """
        INSERT INTO Events (movie_name, park_id, rating, date, closed_captioning)
        VALUES %s;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows, page_size=500)
    conn.commit()

    logger.info(f"  {len(rows):,} events inserted — {skipped:,} skipped")

# =============================================================================
# VERIFY
# =============================================================================
def verify_load(conn):
    logger.info("Verifying loaded tables...")
    tables = ["ZipCodes", "Parks", "Events"]
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                logger.info(f"  {table}: {count:,} rows")
            except Exception as e:
                logger.warning(f"  Could not verify {table}: {e}")
                conn.rollback()

# =============================================================================
# PUBLIC INTERFACE — PRODUCTION (postgres)
# =============================================================================
def run_load_postgres(conn, cleaned_movies: pd.DataFrame, census_data: pd.DataFrame):
    """
    Entry point called by main.py in production mode.
    Load order respects FK constraints:
        1. ZipCodes  (no dependencies)
        2. Parks     (depends on ZipCodes)
        3. Events    (depends on Parks)
    """
    logger.info("=" * 60)
    logger.info("LOAD — writing clean data to postgres")
    logger.info("=" * 60)

    try:
        create_tables(conn)
        ensure_unknown_zip(conn)        # FIX 3 — guarantees '00000' exists before Parks load
        load_zipcodes(conn, census_data)
        ensure_all_movie_zips(conn, cleaned_movies) # ← NEW: backfill any missing zips
        
        park_name_to_id = load_parks(conn, cleaned_movies)
        load_events(conn, cleaned_movies, park_name_to_id)
        verify_load(conn)

        logger.info("=" * 60)
        logger.info("Load complete — all tables written to postgres")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Load failed: {e}")
        conn.rollback()
        raise

# =============================================================================
# PUBLIC INTERFACE — TEST MODE (CSV)
# =============================================================================
def run_load_csv(cleaned_movies: pd.DataFrame, census_df: pd.DataFrame):
    """
    Test mode — writes DataFrames to CSV files in DATA_DIR.
    No postgres connection needed.
    """
    logger.info("=" * 60)
    logger.info("LOAD (TEST MODE) — writing DataFrames to CSV")
    logger.info("=" * 60)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        logger.info("Writing cleaned movies to CSV...")
        cleaned_movies.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"  {len(cleaned_movies):,} rows saved to {OUTPUT_FILE}")
        logger.info(f"  Columns: {cleaned_movies.columns.tolist()}")
        logger.info(f"  Sample:\n{cleaned_movies.head()}")

        logger.info("-" * 60)

        logger.info("Writing census data to CSV...")
        census_df.to_csv(CENSUS_OUTPUT_FILE, index=False)
        logger.info(f"  {len(census_df):,} rows saved to {CENSUS_OUTPUT_FILE}")
        logger.info(f"  Columns: {census_df.columns.tolist()}")
        logger.info(f"  Sample:\n{census_df.head()}")

        logger.info("=" * 60)
        logger.info("LOAD COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise