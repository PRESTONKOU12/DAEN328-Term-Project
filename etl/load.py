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

CENSUS_ZIP_CANDIDATES = ["Zip", "zip_code"]
CENSUS_INCOME_CANDIDATES = ["Median Household Income", "B19013_001E"]
CENSUS_MEDIAN_AGE_CANDIDATES = ["Median Age", "B01002_001E"]
CENSUS_POPULATION_CANDIDATES = ["Total Population", "B01003_001E"]

CENSUS_WHITE_CANDIDATES = ["White Alone Population", "B02001_002E"]
CENSUS_BLACK_CANDIDATES = ["Black Alone Population", "B02001_003E"]
CENSUS_ASIAN_CANDIDATES = ["Asian Alone Population", "B02001_005E"]
CENSUS_OTHER_CANDIDATES = ["Other Race Alone Population", "B02001_007E"]

CENSUS_MARITAL_TOTAL_CANDIDATES = ["Marital Status Universe", "B12001_001E"]
CENSUS_MARRIED_MALE_CANDIDATES = ["Married Male Population", "B12001_004E"]
CENSUS_MARRIED_FEMALE_CANDIDATES = ["Married Female Population", "B12001_013E"]

CENSUS_BIRTH_CANDIDATES = ["Women With Birth Last 12 Months", "B13002_002E"]

CENSUS_EDU_TOTAL_CANDIDATES = ["Education Population Total", "B15003_001E"]
CENSUS_BACHELORS_CANDIDATES = ["Bachelors Degree Population", "B15003_022E"]
CENSUS_MASTERS_CANDIDATES = ["Masters Degree Population", "B15003_023E"]
CENSUS_PROFESSIONAL_CANDIDATES = ["Professional School Degree Population", "B15003_024E"]
CENSUS_DOCTORATE_CANDIDATES = ["Doctorate Degree Population", "B15003_025E"]

# =============================================================================
# TABLE DEFINITIONS
# =============================================================================
CREATE_ZIPCODES_TABLE = """
    CREATE TABLE IF NOT EXISTS ZipCodes (
        zip_code VARCHAR(10) PRIMARY KEY,
        income   INTEGER,
        predominant_race VARCHAR(20),
        pct_married NUMERIC(5,2),
        birth_rate INTEGER,
        edu_rate NUMERIC(5,2),
        median_age NUMERIC(6,2),
        population INTEGER
    );
"""


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _num_or_zero(value) -> float:
    n = pd.to_numeric(value, errors="coerce")
    return float(n) if pd.notna(n) else 0.0


def ensure_zipcodes_schema(conn):
    """
    Add missing ZipCodes columns for newly derived census metrics.
    Safe to run repeatedly.
    """
    alter_sql = [
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS predominant_race VARCHAR(20);",
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS pct_married NUMERIC(5,2);",
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS birth_rate INTEGER;",
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS edu_rate NUMERIC(5,2);",
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS median_age NUMERIC(6,2);",
        "ALTER TABLE ZipCodes ADD COLUMN IF NOT EXISTS population INTEGER;",
    ]
    with conn.cursor() as cur:
        for stmt in alter_sql:
            cur.execute(stmt)
    conn.commit()
    logger.info("ZipCodes schema verified/updated for derived census metrics")

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
    Computes and stores selected derived demographic metrics per zip.

    Supports both readable census names and legacy ACS code columns.
    """
    logger.info("Loading ZipCodes table...")

    zip_col = _pick_col(census_df, CENSUS_ZIP_CANDIDATES)
    income_col = _pick_col(census_df, CENSUS_INCOME_CANDIDATES)
    if not zip_col or not income_col:
        raise KeyError(
            "Census DataFrame must include zip and income columns. "
            f"Found: {census_df.columns.tolist()}"
        )

    median_age_col = _pick_col(census_df, CENSUS_MEDIAN_AGE_CANDIDATES)
    population_col = _pick_col(census_df, CENSUS_POPULATION_CANDIDATES)

    white_col = _pick_col(census_df, CENSUS_WHITE_CANDIDATES)
    black_col = _pick_col(census_df, CENSUS_BLACK_CANDIDATES)
    asian_col = _pick_col(census_df, CENSUS_ASIAN_CANDIDATES)
    other_col = _pick_col(census_df, CENSUS_OTHER_CANDIDATES)

    marital_total_col = _pick_col(census_df, CENSUS_MARITAL_TOTAL_CANDIDATES)
    married_male_col = _pick_col(census_df, CENSUS_MARRIED_MALE_CANDIDATES)
    married_female_col = _pick_col(census_df, CENSUS_MARRIED_FEMALE_CANDIDATES)

    birth_col = _pick_col(census_df, CENSUS_BIRTH_CANDIDATES)

    edu_total_col = _pick_col(census_df, CENSUS_EDU_TOTAL_CANDIDATES)
    bachelors_col = _pick_col(census_df, CENSUS_BACHELORS_CANDIDATES)
    masters_col = _pick_col(census_df, CENSUS_MASTERS_CANDIDATES)
    professional_col = _pick_col(census_df, CENSUS_PROFESSIONAL_CANDIDATES)
    doctorate_col = _pick_col(census_df, CENSUS_DOCTORATE_CANDIDATES)

    # Cast numeric values before aggregation/derivation.
    census_df = census_df.copy()
    census_df[income_col] = pd.to_numeric(census_df[income_col], errors="coerce")
    if median_age_col:
        census_df[median_age_col] = pd.to_numeric(census_df[median_age_col], errors="coerce")
    if population_col:
        census_df[population_col] = pd.to_numeric(census_df[population_col], errors="coerce")

    for col in [
        white_col, black_col, asian_col, other_col,
        marital_total_col, married_male_col, married_female_col,
        birth_col,
        edu_total_col, bachelors_col, masters_col, professional_col, doctorate_col,
    ]:
        if col:
            census_df[col] = pd.to_numeric(census_df[col], errors="coerce")

    # Derived metrics.
    def calc_pred_race(row) -> str:
        races = {
            "White": _num_or_zero(row.get(white_col)) if white_col else 0,
            "Black": _num_or_zero(row.get(black_col)) if black_col else 0,
            "Asian": _num_or_zero(row.get(asian_col)) if asian_col else 0,
            "Other": _num_or_zero(row.get(other_col)) if other_col else 0,
        }
        return max(races, key=races.get)

    def calc_pct_married(row):
        total_married_universe = _num_or_zero(row.get(marital_total_col)) if marital_total_col else 0
        married_count = (
            (_num_or_zero(row.get(married_male_col)) if married_male_col else 0)
            + (_num_or_zero(row.get(married_female_col)) if married_female_col else 0)
        )
        if total_married_universe <= 0:
            return None
        return round((married_count / total_married_universe) * 100, 2)

    def calc_edu_rate(row):
        edu_total = _num_or_zero(row.get(edu_total_col)) if edu_total_col else 0
        bach_plus = (
            (_num_or_zero(row.get(bachelors_col)) if bachelors_col else 0)
            + (_num_or_zero(row.get(masters_col)) if masters_col else 0)
            + (_num_or_zero(row.get(professional_col)) if professional_col else 0)
            + (_num_or_zero(row.get(doctorate_col)) if doctorate_col else 0)
        )
        if edu_total <= 0:
            return None
        return round((bach_plus / edu_total) * 100, 2)

    census_df["predominant_race"] = census_df.apply(calc_pred_race, axis=1)
    census_df["pct_married"] = census_df.apply(calc_pct_married, axis=1)
    census_df["birth_rate"] = (
        pd.to_numeric(census_df[birth_col], errors="coerce").round().astype("Int64")
        if birth_col else pd.Series([pd.NA] * len(census_df), index=census_df.index, dtype="Int64")
    )
    census_df["edu_rate"] = census_df.apply(calc_edu_rate, axis=1)

    if median_age_col:
        census_df["median_age"] = pd.to_numeric(census_df[median_age_col], errors="coerce")
    else:
        census_df["median_age"] = pd.NA

    if population_col:
        census_df["population"] = pd.to_numeric(census_df[population_col], errors="coerce").round().astype("Int64")
    else:
        census_df["population"] = pd.NA

    unique_zips = census_df.groupby(zip_col, as_index=False).agg(
        avg_median_income=(income_col, "mean"),
        predominant_race=("predominant_race", "first"),
        pct_married=("pct_married", "mean"),
        birth_rate=("birth_rate", "mean"),
        edu_rate=("edu_rate", "mean"),
        median_age=("median_age", "mean"),
        population=("population", "mean"),
    )

    rows = []
    for _, row in unique_zips.iterrows():
        zip_code = str(row[zip_col]).strip()
        income   = (
            int(row["avg_median_income"])
            if pd.notna(row["avg_median_income"])
            else None
        )
        predominant_race = str(row["predominant_race"]).strip() if pd.notna(row["predominant_race"]) else None
        pct_married = float(round(row["pct_married"], 2)) if pd.notna(row["pct_married"]) else None
        birth_rate = int(round(row["birth_rate"])) if pd.notna(row["birth_rate"]) else None
        edu_rate = float(round(row["edu_rate"], 2)) if pd.notna(row["edu_rate"]) else None
        median_age = float(round(row["median_age"], 2)) if pd.notna(row["median_age"]) else None
        population = int(round(row["population"])) if pd.notna(row["population"]) else None

        rows.append((
            zip_code,
            income,
            predominant_race,
            pct_married,
            birth_rate,
            edu_rate,
            median_age,
            population,
        ))

    with conn.cursor() as cur:
        for zip_row in rows:
            cur.execute(
                """
                INSERT INTO ZipCodes (
                    zip_code,
                    income,
                    predominant_race,
                    pct_married,
                    birth_rate,
                    edu_rate,
                    median_age,
                    population
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (zip_code) DO UPDATE SET
                    income = EXCLUDED.income,
                    predominant_race = EXCLUDED.predominant_race,
                    pct_married = EXCLUDED.pct_married,
                    birth_rate = EXCLUDED.birth_rate,
                    edu_rate = EXCLUDED.edu_rate,
                    median_age = EXCLUDED.median_age,
                    population = EXCLUDED.population;
                """,
                zip_row
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
        ensure_zipcodes_schema(conn)
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