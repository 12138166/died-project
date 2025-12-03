"""
sync_snowflake_to_dropbox.py

A template script for:
1) Fetching large-scale data from Snowflake in batches (by ID list),
2) Writing each batch to a local Parquet file,
3) Uploading the Parquet file to Dropbox,
4) Recording completed IDs to support resumable runs.

All credentials (Snowflake / Dropbox) are read from environment variables.
"""

import os
import sys
import time
import argparse
import logging
import traceback
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
from tqdm import tqdm
import dropbox

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("snowflake_to_dropbox.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info(f"Snowflake → Dropbox sync script started at {datetime.now():%Y-%m-%d %H:%M:%S}")
logger.info("=" * 80)

# Snowflake configuration (all from environment variables)

SNOWFLAKE_USERNAME   = os.environ.get("SNOWFLAKE_USERNAME")
SNOWFLAKE_PASSWORD   = os.environ.get("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT    = os.environ.get("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_DATABASE   = os.environ.get("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA     = os.environ.get("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE  = os.environ.get("SNOWFLAKE_WAREHOUSE")


def validate_snowflake_config():
    missing = [
        name
        for name, value in [
            ("SNOWFLAKE_USERNAME", SNOWFLAKE_USERNAME),
            ("SNOWFLAKE_PASSWORD", SNOWFLAKE_PASSWORD),
            ("SNOWFLAKE_ACCOUNT", SNOWFLAKE_ACCOUNT),
            ("SNOWFLAKE_DATABASE", SNOWFLAKE_DATABASE),
            ("SNOWFLAKE_SCHEMA", SNOWFLAKE_SCHEMA),
            ("SNOWFLAKE_WAREHOUSE", SNOWFLAKE_WAREHOUSE),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing Snowflake config environment variables: {', '.join(missing)}"
        )


def create_snowflake_engine():
    """Create a SQLAlchemy engine for Snowflake using env credentials."""
    validate_snowflake_config()
    encoded_password = urllib.parse.quote(SNOWFLAKE_PASSWORD, safe="")
    conn_str = (
        f"snowflake://{SNOWFLAKE_USERNAME}:{encoded_password}"
        f"@{SNOWFLAKE_ACCOUNT}/{SNOWFLAKE_DATABASE}/{SNOWFLAKE_SCHEMA}"
        f"?warehouse={SNOWFLAKE_WAREHOUSE}"
    )
    engine = create_engine(
        conn_str,
        pool_size=5,
        max_overflow=10,
        connect_args={"encoding": "utf-8"},
    )
    logger.info("✅ Snowflake engine created.")
    return engine



# Dropbox configuration (all from environment variables)

DROPBOX_APP_KEY        = os.environ.get("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET     = os.environ.get("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN  = os.environ.get("DROPBOX_REFRESH_TOKEN")
# Where to upload files inside the app folder, e.g. "/snowflake_parquet"
DROPBOX_UPLOAD_PATH    = os.environ.get("DROPBOX_UPLOAD_PATH", "/snowflake_parquet")


def validate_dropbox_config():
    missing = [
        name
        for name, value in [
            ("DROPBOX_APP_KEY", DROPBOX_APP_KEY),
            ("DROPBOX_APP_SECRET", DROPBOX_APP_SECRET),
            ("DROPBOX_REFRESH_TOKEN", DROPBOX_REFRESH_TOKEN),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing Dropbox config environment variables: {', '.join(missing)}"
        )


def connect_dropbox_safe():
    """Create a Dropbox client using a refresh token."""
    try:
        validate_dropbox_config()
        dbx = dropbox.Dropbox(
            app_key=DROPBOX_APP_KEY,
            app_secret=DROPBOX_APP_SECRET,
            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
        )
        # Optional: sanity check
        # dbx.users_get_current_account()
        logger.info("✅ Dropbox connection established.")
        return dbx
    except Exception as e:
        logger.error(f"❌ Dropbox auth/connection failed: {e}")
        return None



# ID file helpers (no sensitive content; just generic templates)

def load_ids_from_file(file_path: str) -> List[str]:
    """Load ID list (one ID per line)."""
    if not os.path.exists(file_path):
        logger.error(f"❌ ID file does not exist: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    logger.info(f"✅ Loaded {len(ids):,} IDs from {file_path}")
    return [str(x) for x in ids]


def load_completed_ids(file_path: str) -> List[str]:
    """Load completed IDs from a local TXT file; used for resumable runs."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        done = [line.strip() for line in f if line.strip()]
    logger.info(f"ℹ Loaded {len(done):,} completed IDs from {file_path}")
    return done


def append_ids_to_file(ids: List[str], file_path: str) -> None:
    """Append a list of IDs to a TXT file (one per line)."""
    if not ids:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for id_ in ids:
            f.write(f"{id_}\n")
    logger.info(f"✅ Recorded {len(ids):,} IDs to {file_path}")



# Snowflake query template (generic; replace with your own schema/table)

def generate_query(id_str: str) -> text:
    """
    Generate a SQL query to fetch data for a batch of IDs.

    This is only a TEMPLATE. Replace the FROM/JOIN clauses with your own
    database, schema, and tables. `id_str` is a comma-separated list of quoted IDs,
    e.g. '123','456','789'.
    """
    # --- Example template; replace with your own logic ---
    return text(f"""
    SELECT
        entity_id,
        period_end_date,
        data_item_id,
        data_item_value
    FROM MY_DB.MY_SCHEMA.MY_TABLE
    WHERE entity_id IN ({id_str})
    ORDER BY entity_id, period_end_date, data_item_id
    """)



# Dropbox upload helper
def upload_to_dropbox(dbx: dropbox.Dropbox, local_path: str, dropbox_folder: str) -> bool:
    """Upload a local file to Dropbox under dropbox_folder."""
    if not dbx:
        return False

    file_name = os.path.basename(local_path)
    dropbox_path = os.path.join(dropbox_folder, file_name).replace("\\", "/")

    logger.info(f"🚀 Uploading {file_name} → Dropbox: {dropbox_path}")
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        dbx.files_upload(
            data,
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite,
        )
        logger.info(f"✅ Uploaded to Dropbox: {dropbox_path}")
        return True
    except dropbox.exceptions.AuthError:
        logger.error("❌ Dropbox authentication error. Please check your tokens.")
        return False
    except Exception as e:
        logger.error(f"❌ Dropbox upload failed: {e}")
        return False


# Batch processing logic

def process_batch(
    batch_ids: List[str],
    batch_index: int,
    engine,
    dbx: dropbox.Dropbox,
    output_dir: str,
) -> int:
    """
    Fetch data from Snowflake for a list of IDs, save to Parquet, upload to Dropbox.

    Returns:
        number of rows fetched in this batch.
    """
    start_time = time.time()
    id_str = ",".join(f"'{x}'" for x in batch_ids)
    logger.info(f"--- Processing batch #{batch_index} ({len(batch_ids)} IDs) ---")

    os.makedirs(output_dir, exist_ok=True)
    final_parquet_path = os.path.join(
        output_dir,
        f"batch_{batch_index}_{len(batch_ids)}_ids.parquet",
    )

    total_rows = 0

    try:
        with engine.connect() as conn:
            sql = generate_query(id_str)
            # chunk_size can be tuned according to memory
            chunk_size = 100000
            chunks = pd.read_sql(sql, conn, chunksize=chunk_size)

            dfs = []
            for chunk_idx, df_chunk in enumerate(chunks):
                if df_chunk.empty:
                    continue
                rows = len(df_chunk)
                total_rows += rows
                dfs.append(df_chunk)
                logger.info(
                    f"  Batch #{batch_index}, chunk {chunk_idx}: {rows:,} rows"
                )

            if total_rows == 0:
                logger.info(f"ℹ Batch #{batch_index} returned no rows.")
                return 0

            # Concatenate all chunks & write to Parquet
            full_df = pd.concat(dfs, ignore_index=True)
            full_df.to_parquet(final_parquet_path, compression="snappy")
            logger.info(
                f"✅ Wrote Parquet file: {final_parquet_path} "
                f"({os.path.getsize(final_parquet_path) / (1024**2):.2f} MB)"
            )

        # Upload to Dropbox
        upload_success = upload_to_dropbox(dbx, final_parquet_path, DROPBOX_UPLOAD_PATH)

        # Optionally remove local file after successful upload
        if upload_success:
            os.remove(final_parquet_path)
            logger.info(f"🧹 Removed local file: {final_parquet_path}")

        elapsed = time.time() - start_time
        logger.info(
            f"Batch #{batch_index} finished: {total_rows:,} rows in {elapsed:.1f} seconds"
        )
        return total_rows

    except Exception as e:
        logger.error(
            f"❌ Batch #{batch_index} failed: {e}\n{traceback.format_exc()}"
        )
        # Keep local file (if any) for debugging
        return 0



# Main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync Snowflake data to Dropbox in batches by ID list."
    )
    parser.add_argument(
        "--id_file",
        type=str,
        required=True,
        help="Path to a text file with one ID per line.",
    )
    parser.add_argument(
        "--completed_ids_file",
        type=str,
        default="completed_ids.txt",
        help="Path to a local TXT file recording completed IDs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_parquet",
        help="Local directory for temporary Parquet files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of IDs per batch.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the ID list (for manual sharding).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) in the ID list.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Connect Dropbox
    dbx = connect_dropbox_safe()
    if not dbx:
        logger.error("Dropbox connection not available. Exiting.")
        return

    # Load ID list
    all_ids = load_ids_from_file(args.id_file)
    if not all_ids:
        logger.error("ID list is empty. Exiting.")
        return

    # Load completed IDs for resumable run
    completed_ids = set(load_completed_ids(args.completed_ids_file))

    # Filter remaining IDs
    remaining = [id_ for id_ in all_ids if id_ not in completed_ids]

    # Apply slice [start:end]
    remaining_slice = remaining[args.start : args.end]
    logger.info(
        f"Total IDs: {len(all_ids):,} | Completed: {len(completed_ids):,} | "
        f"Remaining (un-sliced): {len(remaining):,} | "
        f"This run: {len(remaining_slice):,} (from index {args.start} to {args.end})"
    )

    if not remaining_slice:
        logger.info("No IDs left to process in the specified range. Exiting.")
        return

    # Create Snowflake engine
    engine = create_snowflake_engine()

    # Prepare batches
    batch_size = args.batch_size
    batches = [
        remaining_slice[i : i + batch_size]
        for i in range(0, len(remaining_slice), batch_size)
    ]
    logger.info(f"Total batches in this run: {len(batches)} (batch_size={batch_size})")

    total_rows = 0
    pbar = tqdm(total=len(batches), desc="Batches", unit="batch")

    for batch_index, batch_ids in enumerate(batches, start=1):
        rows = process_batch(
            batch_ids=batch_ids,
            batch_index=batch_index,
            engine=engine,
            dbx=dbx,
            output_dir=args.output_dir,
        )
        total_rows += rows

        # Mark IDs in this batch as completed if processing succeeded
        if rows > 0:
            append_ids_to_file(batch_ids, args.completed_ids_file)

        pbar.update(1)
        pbar.set_postfix(rows=f"{total_rows:,}")

    pbar.close()
    logger.info("=" * 80)
    logger.info(f"All done. Total rows processed in this run: {total_rows:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
