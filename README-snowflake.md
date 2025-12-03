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
