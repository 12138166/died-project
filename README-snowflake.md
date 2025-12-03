# Snowflake → Dropbox Sync (Batch ETL Template)

This repository provides a **template script** for syncing data from
**Snowflake** to **Dropbox** in batches, using an ID list:

1. Read a list of entity IDs from a text file.
2. For each batch of IDs, run a SQL query against Snowflake.
3. Save the result as a local Parquet file.
4. Upload the Parquet file to Dropbox.
5. Record completed IDs to a local TXT file to support **resumable runs**.

All Snowflake and Dropbox credentials are read from **environment variables**,
so no secrets are hard-coded in the script.

---

## 1. Requirements

- Python 3.8+ (other versions may also work)
- A Snowflake account with read access to the desired tables
- A Dropbox app (App folder or full Dropbox) with a **refresh token**
- Python packages:
  - `pandas`
  - `sqlalchemy`
  - `tqdm`
  - `dropbox`
  - `pyarrow` (for Parquet)
  - `snowflake-sqlalchemy` / `snowflake-connector-python` (depending on your setup)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Environment Variables

Set the following environment variables before running the script.

Snowflake
```bash
export SNOWFLAKE_USERNAME="your_username"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_ACCOUNT="your_account_region"
export SNOWFLAKE_DATABASE="YOUR_DATABASE"
export SNOWFLAKE_SCHEMA="YOUR_SCHEMA"
export SNOWFLAKE_WAREHOUSE="YOUR_WAREHOUSE"
```

Dropbox
```bash
export DROPBOX_APP_KEY="your_app_key"
export DROPBOX_APP_SECRET="your_app_secret"
export DROPBOX_REFRESH_TOKEN="your_refresh_token"
# Optional: target folder inside your app folder
export DROPBOX_UPLOAD_PATH="/snowflake_parquet"
```

Security note:
Do not hard-code any of these credentials in the script.
Keep them in your shell profile, a .env file (not checked into git), or in a secure secrets manager.



## 3. Preparing the ID List
ave it as all_ids.txt (or any name you like).


## 4. Customizing the SQL Query

In sync_snowflake_to_dropbox.py, the function generate_query(id_str) is a template:
```bash
def generate_query(id_str: str) -> text:
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
```

Replace MY_DB.MY_SCHEMA.MY_TABLE and the selected columns with your own
database, schema, and table names.
The variable id_str is a comma-separated list of quoted IDs, e.g.:
```sql
WHERE entity_id IN ('10001','10002','10003')
```

## 5. Usage

Basic usage:
```bash
python sync_snowflake_to_dropbox.py \
  --id_file all_ids.txt \
  --completed_ids_file completed_ids.txt \
  --output_dir output_parquet \
  --batch_size 50
```
Arguments

--id_file
Path to the TXT file containing the full ID list (one per line).

--completed_ids_file
Path to a TXT file used to record IDs that have been successfully processed.
This file is used for resumable runs.
Default: completed_ids.txt.

--output_dir
Local directory where temporary Parquet files are written before uploading.
Default: output_parquet.

--batch_size
Number of IDs per Snowflake query. Default: 50.

--start, --end
Optional indices for slicing the ID list, useful for manual sharding
(e.g., when running the script on multiple machines).

Example: process IDs from index 1000 to 2000:
```bash
python sync_snowflake_to_dropbox.py \
  --id_file all_ids.txt \
  --completed_ids_file completed_ids.txt \
  --output_dir output_parquet \
  --batch_size 50 \
  --start 1000 \
  --end 2000
```


## 6. How It Works (High-Level)

- Load IDs
  - Read all IDs from --id_file.
  - Read completed IDs from --completed_ids_file.
  - Filter out already completed IDs.

- Batching
  - Split remaining IDs into batches of size --batch_size.

- For each batch
  - Build a SQL query with WHERE entity_id IN (...).
  - Fetch data from Snowflake in chunks (to avoid memory issues).
  - Concatenate chunks and write a Parquet file locally.
  - Upload the Parquet file to Dropbox under DROPBOX_UPLOAD_PATH.
  - Record batch IDs to --completed_ids_file.

- Resumable
  - If the script stops halfway, re-running it will skip IDs already
recorded in --completed_ids_file, and continue with the remaining ones.

## 7. Notes and Extensions
- You can extend the script to:
  - Add resource monitoring (CPU/memory logs).
  - Support multiple ID sources (e.g. "all" vs "missing" lists).
  - Split into modules (e.g. config_snowflake.py, config_dropbox.py).

- The current template is intentionally generic and does not expose
any real account details, database names, or ID values.


## 8. Disclaimer

This repository is intended as a template for building your own ETL
pipeline between Snowflake and Dropbox.

Always make sure:
- Your use of data complies with your data provider’s terms and conditions.
- You keep all credentials in a secure place and never commit them to git.











