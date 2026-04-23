#!/usr/bin/env python3
"""
Movies in the Park — ETL Orchestrator
Toggle between test mode (CSV) and production mode (postgres)
by swapping the load call as noted below.
"""
import logging
import time
import os
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_HOST     = os.environ.get("DB_HOST", "postgres")
DB_PORT     = os.environ.get("DB_PORT", "5432")
DB_NAME     = os.environ.get("DB_NAME", "movies_db")
DB_USER     = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def get_connection(retries: int = 10, delay: int = 5):
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} "
                f"(attempt {attempt}/{retries})..."
            )
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=10,
            )
            logger.info("Successfully connected to PostgreSQL")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Connection failed: {e}")
            if attempt < retries:
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All connection attempts exhausted")
                raise


def main():
    logger.info("=" * 60)
    logger.info("MOVIES IN THE PARK — ETL PIPELINE")
    logger.info("=" * 60)

    conn = None

    try:
        conn = get_connection()

        # ------------------------------------------------------------------
        # PHASE 1 — EXTRACT
        # Returns (movies_df, census_df) as separate DataFrames
        # ------------------------------------------------------------------
        logger.info("PHASE 1 — EXTRACT")
        logger.info("-" * 60)
        from extract import run_extract
        movies_df, census_df = run_extract()
        logger.info(f"Extract complete — {len(movies_df):,} movie rows, {len(census_df):,} census rows")

        # ------------------------------------------------------------------
        # PHASE 2 — TRANSFORM
        # Receives movies_df, returns clean_movies_df
        # Census data is not transformed — passed directly to load
        # ------------------------------------------------------------------
        logger.info("PHASE 2 — TRANSFORM")
        logger.info("-" * 60)
        from transform import run_transform
        clean_movies_df = run_transform(movies_df)
        logger.info(f"Transform complete — {len(clean_movies_df):,} clean movie rows")

        # ------------------------------------------------------------------
        # PHASE 3 — LOAD
        # ------------------------------------------------------------------
        logger.info("PHASE 3 — LOAD")
        logger.info("-" * 60)
        from load import run_load_postgres
        run_load_postgres(conn, clean_movies_df, census_df)

        # ------------------------------------------------------------------
        # To switch to CSV test mode swap the above two lines for:
        #
        # from load import run_load_csv
        # run_load_csv(clean_movies_df, census_df)
        #
        # No postgres connection is used in CSV mode
        # ------------------------------------------------------------------

        logger.info("=" * 60)
        logger.info("ETL PIPELINE COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()