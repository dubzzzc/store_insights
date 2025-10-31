#!/usr/bin/env python3
r"""
VFP DBF → RDS Uploader
- Lists .DBF tables from a chosen ksv\data folder
- Infers schema from DBF and auto-creates tables in RDS (SQL Server or MySQL)
- Bulk inserts data with batching
- Supports GUI mode (Tkinter or DearPyGui) and headless mode via --config YAML
- Stores config next to \ksv\ by default (grandma-proof)
- Supports multi-store profiles
- Fresh-load safety: drop_recreate option to avoid duplicates
- Delta/Incremental sync: Only sync new records since last sync (per store/table)
- Auto-sync: Periodic automatic syncing with configurable intervals

Prereqs (Windows):
  pip install dbfread pyyaml pyodbc mysql-connector-python dearpygui

Example config.yaml (single or multi-profile):
---
# Single-profile format (no 'profiles' key):
rds:
  engine: mysql
  server: your-ec2-instance.amazonaws.com
  port: 3306
  database: "your_database"
  username: your_user
  password: your_password_here
source:
  folder: "C:/ksv/data"
  include: ["inv.dbf", "stk.dbf"]   # omit for all DBFs
load:
  drop_recreate: false         # set to false to enable delta sync
  truncate_before_load: false
  batch_size: 1000
  table_prefix: ""
  table_suffix: ""
  coerce_lowercase_names: true
  nvarchar_default: 255
delta_sync:
  enabled: true                # Enable incremental sync
  date_field: "date"            # Field name in DBF to use for date filtering (case-insensitive)
                                # Common fields: "date", "updated", "created", "timestamp"
  auto_sync_interval_seconds: 3600  # Auto-sync every hour (0 = disabled)

# Multi-profile format:
profiles:
  store6885:
    rds:
      engine: mysql
      server: ec2-instance.amazonaws.com
      port: 3306
      database: SpiritsDemo_6885
      username: store6885_ingest
      password: your_password_here
    source:
      folder: "C:/ksv/data_6885"
    load:
      drop_recreate: false
      truncate_before_load: false
      batch_size: 1000
      table_prefix: ""
      table_suffix: "_6885"
      coerce_lowercase_names: true
    delta_sync:
      enabled: true
      date_field: "date"
      auto_sync_interval_seconds: 1800  # 30 minutes

Usage:
  python vfp_dbf_to_rdsv2.py --gui                    # Launch GUI
  python vfp_dbf_to_rdsv2.py --config path/to/config.yaml  # Run once
  python vfp_dbf_to_rdsv2.py --auto-sync              # Run auto-sync (periodic)
  python vfp_dbf_to_rdsv2.py --profile store6885      # Use specific profile
"""

import os
import re
import sys
import glob
import json
from urllib import request as urlrequest
from urllib import parse as urlparse
import yaml
import getpass
import time
import threading
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Iterable, Tuple, Optional

# ---------------- Config persistence helpers ----------------
APP_NAME = "StoreInsights"
CONFIG_NAME = "vfp_uploader.yaml"
SYNC_TRACKING_NAME = "vfp_sync_tracking.yaml"

def debug_config_path(msg: str, path: str):
    try:
        print(f"[CFG] {msg}: {path}")
    except Exception:
        pass

def find_ksv_root(start: Optional[str] = None) -> Optional[str]:
    r"""
    Try to locate a folder literally named 'ksv' (C:\ksv or ancestor).
    Priority:
      1) current working dir or its parents
      2) script dir or its parents
      3) hint file 'ksv_path.txt' next to this script
    """
    import pathlib
    candidates = []
    if start:
        candidates.append(pathlib.Path(start).resolve())
    candidates.append(pathlib.Path.cwd())
    candidates.append(pathlib.Path(__file__).resolve().parent)

    for base in candidates:
        p = base
        for _ in range(6):
            if p.name.lower() == "ksv":
                return str(p)
            p = p.parent

    hint = pathlib.Path(__file__).resolve().parent / "ksv_path.txt"
    if hint.exists():
        txt = hint.read_text(encoding="utf-8").strip()
        if txt and os.path.isdir(txt) and os.path.basename(txt).lower() == "ksv":
            return txt
    return None

def ksv_config_path(ksv_root: str) -> str:
    return os.path.join(ksv_root, CONFIG_NAME)

def default_config_path() -> str:
    # Prefer local \ksv\vfp_uploader.yaml
    ksv = find_ksv_root()
    if ksv:
        path = ksv_config_path(ksv)
        return path
    # Fallback to AppData
    base = os.getenv("APPDATA") or os.path.expanduser("~/.config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, CONFIG_NAME)

def save_config(cfg: dict, path: Optional[str] = None):
    path = path or default_config_path()
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    debug_config_path("Saved config", path)
    return path

def load_config(path: Optional[str] = None) -> dict:
    path = path or default_config_path()
    debug_config_path("Loading config", path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def default_sync_tracking_path() -> str:
    """Get path to sync tracking file (stores last sync timestamps per table/store)."""
    ksv = find_ksv_root()
    if ksv:
        return os.path.join(ksv, SYNC_TRACKING_NAME)
    base = os.getenv("APPDATA") or os.path.expanduser("~/.config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, SYNC_TRACKING_NAME)

def load_sync_tracking(path: Optional[str] = None) -> dict:
    """Load sync tracking data (last sync timestamps per table)."""
    path = path or default_sync_tracking_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception:
            return {}
    return {}

def save_sync_tracking(tracking: dict, path: Optional[str] = None):
    """Save sync tracking data."""
    path = path or default_sync_tracking_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(tracking, f, sort_keys=False)
    debug_config_path("Saved sync tracking", path)

def resolve_profile(cfg: dict, profile: Optional[str]) -> dict:
    """
    Return a single-profile dict with keys rds/source/load.
    If 'profiles' exists, require --profile and return that entry.
    Otherwise treat cfg as single-profile.
    """
    if 'profiles' in cfg:
        if not profile:
            raise RuntimeError("Config contains multiple profiles. Use --profile <name>.")
        if profile not in cfg['profiles']:
            raise RuntimeError(f"Profile '{profile}' not found in config.")
        return cfg['profiles'][profile]
    return cfg

def ensure_password(cfg: dict) -> str:
    """Read password directly from YAML config."""
    r = cfg.get('rds', {})
    pw = r.get('password')
    if not pw:
        raise RuntimeError("No password specified in config. Please add 'password' to rds section in YAML.")
    return pw
# ---------- Admin backend creds fetch ----------
def fetch_admin_creds(base_url: str, api_key: str, store_id: str) -> dict:
    """Fetch per-store DB credentials from admin backend.

    Returns a dict with keys: engine, host, port, database, username, password, schema
    """
    if not base_url:
        raise RuntimeError("Admin base_url is required when admin_api.enabled is true")
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    url = f"{base_url}/uploader/creds?{urlparse.urlencode({'store_id': store_id})}"
    req = urlrequest.Request(url)
    req.add_header('X-API-Key', api_key or '')
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Admin creds fetch failed: HTTP {resp.status}")
            data = json.loads(resp.read().decode('utf-8'))
            # Basic shape check
            for k in ("host","port","database","username","password"):
                if k not in data:
                    raise RuntimeError(f"Admin creds missing field: {k}")
            return data
    except Exception as e:
        raise RuntimeError(f"Admin creds fetch error: {e}")

# --- Only scan these DBFs ---
ALLOWED_BASES = {
    "cat","cus","emp","glb","inv","jnh","jnl","pod","poh","prc","sll","stk","upc","vnd","timeclock","hst"
}

def is_allowed_dbf(path: str) -> bool:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    return base in ALLOWED_BASES

def list_allowed_dbfs(folder: str, include: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of *.dbf in folder limited to ALLOWED_BASES.
       If include list is provided, still filter to ALLOWED_BASES.
    """
    if include:
        files = [os.path.join(folder, f) for f in include]
    else:
        files = glob.glob(os.path.join(folder, "*.dbf"))
    return sorted([p for p in files if is_allowed_dbf(p)])


from dbfread import DBF

def open_dbf(path: str, encodings=("latin-1", "cp1252", "cp437", "utf-8")) -> DBF:
    last_err = None
    for enc in encodings:
        try:
            return DBF(path, encoding=enc, load=False)
        except Exception as e:
            last_err = e
    raise last_err

# Optional imports depending on target engine
try:
    import pyodbc  # SQL Server
except Exception:
    pyodbc = None

try:
    import mysql.connector  # MySQL option
except Exception:
    mysql = None

RESERVED = set(
    """
    add all alter and any as asc authorization backup begin between break browse bulk by cascade check cluster column commit
    constraint contains contains_table continue convert create cross current_date current_time current_timestamp cursor
    database dbcc deallocate declare default delete deny desc disk distinct distributed double drop else end errlvl escape except exec
    execute exists exit external fetch file fillfactor for foreign freetext freetexttable from full function goto grant group having holdlock
    identity identity_insert identitycol if in index inner insert intersect into is join key kill left like lineno load merge national nocheck
    nocount nolock nonclustered not null nullif of off offsets on open opendatasource openquery openrowset openxml option or order outer over
    percent plan precision primary print proc procedure public raiserror read readtext reconfigure references replication restore restrict return
    revert right rollback rowcount rowguidcol rule save schema securityaudit select session_user set setuser shutdown some statistics system_user
    table tablesample textsize then to top tran transaction trigger truncate tsequal union unique update updatetext use user values varying view waitfor
    when where while with writetext
    """.split()
)

# ---------- DBF → SQL type mapping ----------

def safe_sql_name(name: str, coerce_lower: bool = True) -> str:
    n = name.strip().replace(" ", "_")
    n = re.sub(r"[^A-Za-z0-9_]", "_", n)
    if coerce_lower:
        n = n.lower()
    if n.lower() in RESERVED or re.match(r"^[0-9]", n):
        n = f"_{n}"
    return n

def map_dbf_field_to_sql(field, engine: str = "mysql", nvarchar_default: int = 255) -> str:
    ftype = field.type  # 'C', 'N', 'F', 'D', 'T', 'L', 'M', 'I', 'B'
    length = getattr(field, 'length', None) or getattr(field, 'size', None) or nvarchar_default
    decimals = getattr(field, 'decimal_count', 0)

    if engine == "mssql":
        if ftype == 'C':
            return f"NVARCHAR({length if length and length <= 4000 else 'MAX'})"
        if ftype in ('N', 'F'):
            if decimals and decimals > 0:
                precision = max(length or 18, decimals + 1)
                scale = decimals
                precision = min(38, precision)
                return f"DECIMAL({precision},{scale})"
            else:
                return "INT" if (length or 10) <= 9 else ("BIGINT" if (length or 19) <= 18 else "DECIMAL(38,0)")
        if ftype == 'I':
            return "INT"
        if ftype == 'B':
            return "FLOAT"
        if ftype == 'D':
            return "DATE"
        if ftype in ('T', '@'):
            return "DATETIME2(3)"
        if ftype == 'L':
            return "BIT"
        if ftype == 'M':
            return "NVARCHAR(MAX)"
        return f"NVARCHAR({nvarchar_default})"

    # mysql (default)
    if ftype == 'C':
        return f"VARCHAR({min(length or nvarchar_default, 65535)})"
    if ftype in ('N', 'F'):
        if decimals and decimals > 0:
            precision = max(length or 18, decimals + 1)
            scale = decimals
            return f"DECIMAL({precision},{scale})"
        else:
            return "INT" if (length or 10) <= 9 else ("BIGINT" if (length or 19) <= 18 else "DECIMAL(38,0)"
                   )
    if ftype == 'I':
        return "INT"
    if ftype == 'B':
        return "DOUBLE"
    if ftype == 'D':
        return "DATE"
    if ftype in ('T', '@'):
        return "DATETIME"
    if ftype == 'L':
        return "TINYINT(1)"
    if ftype == 'M':
        return "LONGTEXT"
    return f"VARCHAR({nvarchar_default})"

# ---------- DDL generation & schema reconciliation ----------

def build_create_table_sql(table: str, fields, engine: str, schema: str = None) -> str:
    col_defs = []
    for f in fields:
        col_name = safe_sql_name(f.name)
        col_type = map_dbf_field_to_sql(f, engine)
        col_defs.append(f"[{col_name}] {col_type}" if engine == 'mssql' else f"`{col_name}` {col_type}")

    if engine == 'mssql':
        target = f"[{schema}].[{safe_sql_name(table)}]" if schema else f"[{safe_sql_name(table)}]"
        ddl = f"CREATE TABLE {target} (\n  [__id] INT IDENTITY(1,1) PRIMARY KEY,\n  {',\n  '.join(col_defs)}\n)"
    else:
        target = f"`{safe_sql_name(table)}`"
        ddl = (
            f"CREATE TABLE {target} (\n"
            f"  `__id` BIGINT NOT NULL AUTO_INCREMENT,\n  {',\n  '.join(col_defs)},\n"
            f"  PRIMARY KEY(`__id`)\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        )
    return ddl

# ---------- Connections ----------

def connect_mssql(server: str, database: str, username: str, password: str, port: int = 1433):
    if pyodbc is None:
        raise RuntimeError("pyodbc not installed. pip install pyodbc")
    driver = 'ODBC Driver 17 for SQL Server'
    conn_str = f"DRIVER={{{{}}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};"
    # Put driver name inside braces with value:
    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};"
    conn = pyodbc.connect(conn_str, autocommit=False)
    try:
        conn.fast_executemany = True
    except Exception:
        pass
    return conn

def connect_mysql(host: str, database: str, username: str, password: str, port: int = 3306):
    if mysql is None:
        raise RuntimeError("mysql-connector-python not installed. pip install mysql-connector-python")
    return mysql.connector.connect(host=host, user=username, password=password, database=database, port=port)

# ---------- Helpers ----------

def table_exists(conn, engine: str, table: str, schema: str = None) -> bool:
    cur = conn.cursor()
    if engine == 'mssql':
        if schema:
            cur.execute("""
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """, (schema, safe_sql_name(table)))
        else:
            cur.execute("""
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = ?
            """, (safe_sql_name(table),))
        row = cur.fetchone()
        return bool(row)
    else:
        cur.execute("SHOW TABLES LIKE %s", (safe_sql_name(table),))
        return bool(cur.fetchone())

def existing_columns(conn, engine: str, table: str, schema: str = None) -> List[str]:
    cur = conn.cursor()
    if engine == 'mssql':
        if schema:
            cur.execute("""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """, (schema, safe_sql_name(table)))
        else:
            cur.execute("""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
            """, (safe_sql_name(table),))
        return [r[0] for r in cur.fetchall()]
    else:
        cur.execute(f"SHOW COLUMNS FROM `{safe_sql_name(table)}`")
        return [r[0] for r in cur.fetchall()]

def choose_column_name(dbf_name: str, existing: List[str]) -> str:
    wanted = safe_sql_name(dbf_name)
    existing_lower = {c.lower(): c for c in existing}
    if wanted.lower() in existing_lower:
        return existing_lower[wanted.lower()]
    raw = re.sub(r"[^A-Za-z0-9_]", "_", dbf_name.strip())
    if raw.lower() in existing_lower:
        return existing_lower[raw.lower()]
    no_lead_us = wanted[1:] if wanted.startswith('_') else wanted
    if no_lead_us.lower() in existing_lower:
        return existing_lower[no_lead_us.lower()]
    return wanted

def add_missing_column(conn, engine: str, table: str, col_name: str, field, schema: str = None):
    col_type = map_dbf_field_to_sql(field, engine)
    if engine == 'mssql':
        target = f"[{schema}].[{safe_sql_name(table)}]" if schema else f"[{safe_sql_name(table)}]"
        sql = f"ALTER TABLE {target} ADD [{col_name}] {col_type}"
    else:
        target = f"`{safe_sql_name(table)}`"
        sql = f"ALTER TABLE {target} ADD `{col_name}` {col_type}"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()

def drop_table(conn, engine: str, table: str, schema: str = None):
    cur = conn.cursor()
    if engine == 'mssql':
        target = f"[{schema}].[{safe_sql_name(table)}]" if schema else f"[{safe_sql_name(table)}]"
        cur.execute(f"IF OBJECT_ID(N'{target}', N'U') IS NOT NULL DROP TABLE {target}")
    else:
        target = f"`{safe_sql_name(table)}`"
        cur.execute(f"DROP TABLE IF EXISTS {target}")
    conn.commit()

def ensure_table(conn, engine: str, table: str, fields, schema: str = None, recreate: bool = False):
    if recreate and table_exists(conn, engine, table, schema):
        drop_table(conn, engine, table, schema)
    if not table_exists(conn, engine, table, schema):
        ddl = build_create_table_sql(table, fields, engine, schema)
        cur = conn.cursor()
        cur.execute(ddl)
        conn.commit()
        return
    if not recreate:
        existing = existing_columns(conn, engine, table, schema)
        for f in fields:
            chosen = choose_column_name(f.name, existing)
            if chosen.lower() not in {c.lower() for c in existing}:
                add_missing_column(conn, engine, table, safe_sql_name(f.name), f, schema)
                existing.append(safe_sql_name(f.name))

def truncate_table(conn, engine: str, table: str, schema: str = None):
    cur = conn.cursor()
    if engine == 'mssql':
        target = f"[{schema}].[{safe_sql_name(table)}]" if schema else f"[{safe_sql_name(table)}]"
        cur.execute(f"TRUNCATE TABLE {target}")
    else:
        target = f"`{safe_sql_name(table)}`"
        cur.execute(f"TRUNCATE TABLE {target}")
    conn.commit()

def coerce_value(v):
    if isinstance(v, bytes):
        try:
            return v.decode('utf-8', errors='replace')
        except Exception:
            return v.decode('latin-1', errors='replace')
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (datetime, date)):
        return v
    if isinstance(v, bool):
        return 1 if v else 0
    return v

def iter_dbf_rows(dbf_path: str, date_field: Optional[str] = None, since_date: Optional[datetime] = None) -> Tuple[List[str], Iterable[List[Any]]]:
    """
    Iterate DBF rows, optionally filtering by date field.
    
    Args:
        dbf_path: Path to DBF file
        date_field: Field name to use for date filtering (case-insensitive)
        since_date: Only include records with date_field >= since_date
    """
    table = open_dbf(dbf_path)
    field_names = [f.name for f in table.fields]
    
    # Find date field index (case-insensitive)
    date_field_idx = None
    if date_field and since_date:
        for i, name in enumerate(field_names):
            if name.lower() == date_field.lower():
                date_field_idx = i
                break
    
    def gen():
        for rec in table:
            row = [coerce_value(rec.get(name)) for name in field_names]
            
            # Apply date filter if specified
            if date_field_idx is not None and since_date:
                date_val = row[date_field_idx]
                # Handle None, date, datetime, or string dates
                if date_val is None:
                    continue
                if isinstance(date_val, str):
                    try:
                        # Try parsing common date formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                            try:
                                date_val = datetime.strptime(date_val, fmt)
                                break
                            except:
                                continue
                        if isinstance(date_val, str):
                            continue  # Couldn't parse
                    except:
                        continue
                if isinstance(date_val, date) and not isinstance(date_val, datetime):
                    date_val = datetime.combine(date_val, datetime.min.time())
                if not isinstance(date_val, datetime):
                    continue
                # Compare dates (ignore time component if since_date is date-only)
                if date_val < since_date:
                    continue
            
            yield row
    return field_names, gen()

def build_insert_sql(table: str, col_names: List[str], engine: str, schema: str = None, existing: List[str] = None) -> Tuple[str, List[str]]:
    if existing is None:
        dest_cols = [safe_sql_name(c) for c in col_names]
    else:
        dest_cols = [choose_column_name(c, existing) for c in col_names]

    if engine == 'mssql':
        target = f"[{schema}].[{safe_sql_name(table)}]" if schema else f"[{safe_sql_name(table)}]"
        cols = ", ".join(f"[{c}]" for c in dest_cols)
        placeholders = ", ".join(["?"] * len(dest_cols))
        return f"INSERT INTO {target} ({cols}) VALUES ({placeholders})", dest_cols
    else:
        target = f"`{safe_sql_name(table)}`"
        cols = ", ".join(f"`{c}`" for c in dest_cols)
        placeholders = ", ".join(["%s"] * len(dest_cols))
        return f"INSERT INTO {target} ({cols}) VALUES ({placeholders})", dest_cols

def bulk_insert(conn, engine: str, table: str, dbf_path: str, batch_size: int = 1000, schema: str = None, recreate: bool = False, date_field: Optional[str] = None, since_date: Optional[datetime] = None) -> Tuple[int, int]:
    """
    Bulk insert DBF data into RDS table.
    
    Args:
        date_field: Field name to use for delta sync filtering
        since_date: Only insert records newer than this date (delta sync)
    """
    dbf_obj = open_dbf(dbf_path)
    ensure_table(conn, engine, table, dbf_obj.fields, schema, recreate=recreate)

    cols_existing = existing_columns(conn, engine, table, schema)
    col_names, row_iter = iter_dbf_rows(dbf_path, date_field=date_field, since_date=since_date)
    insert_sql, _dest_cols = build_insert_sql(table, col_names, engine, schema, cols_existing)

    cur = conn.cursor()
    buf = []
    inserted = 0
    batches = 0

    for row in row_iter:
        buf.append(row)
        if len(buf) >= batch_size:
            cur.executemany(insert_sql, buf)
            inserted += len(buf)
            batches += 1
            buf.clear()
    if buf:
        cur.executemany(insert_sql, buf)
        inserted += len(buf)
        batches += 1

    conn.commit()
    return inserted, batches

# ---------- Headless mode ----------

def run_headless(cfg_path: Optional[str] = None, profile: Optional[str] = None, auto_sync: bool = False):
    raw_cfg = load_config(cfg_path) if cfg_path else load_config()
    cfg = resolve_profile(raw_cfg, profile)

    engine = cfg['rds'].get('engine', 'mysql')
    server = cfg['rds']['server']
    port = int(cfg['rds'].get('port', 3306 if engine=='mysql' else 1433))
    database = cfg['rds']['database']
    username = cfg['rds']['username']
    password = ensure_password(cfg)
    schema = cfg['rds'].get('schema') if engine=='mssql' else None

    # Admin API override
    admin_cfg = cfg.get('admin_api', {})
    if admin_cfg.get('enabled'):
        creds = fetch_admin_creds(
            admin_cfg.get('base_url', ''),
            admin_cfg.get('api_key', ''),
            str(admin_cfg.get('store_id', '')),
        )
        engine = creds.get('engine', engine)
        server = creds['host']
        port = int(creds.get('port', port))
        database = creds['database']
        username = creds['username']
        password = creds['password']
        schema = creds.get('schema') if engine == 'mssql' else None

    folder = cfg['source']['folder']
    include = cfg['source'].get('include')

    drop_recreate = bool(cfg['load'].get('drop_recreate', False))
    trunc = bool(cfg['load'].get('truncate_before_load', False))
    batch = int(cfg['load'].get('batch_size', 1000))
    tpref = cfg['load'].get('table_prefix', '')
    tsuff = cfg['load'].get('table_suffix', '')
    lower = bool(cfg['load'].get('coerce_lowercase_names', True))

    # Delta sync settings
    delta_cfg = cfg.get('delta_sync', {})
    delta_enabled = bool(delta_cfg.get('enabled', False)) and not drop_recreate
    date_field = delta_cfg.get('date_field')  # Field name to use for filtering (e.g., 'date', 'updated', 'created')
    
    # Always load sync tracking to track last sync time
    sync_tracking = load_sync_tracking()
    
    # Create tracking key from profile + (store_id when admin) + database + folder
    profile_key = profile or 'default'
    if admin_cfg.get('enabled') and str(admin_cfg.get('store_id', '')).strip():
        tracking_key = f"{profile_key}|{admin_cfg.get('store_id')}|{database}|{folder}"
    else:
        tracking_key = f"{profile_key}|{database}|{folder}"

    # Warn if delta enabled but no date field specified
    if delta_enabled and not date_field:
        print("WARNING: Delta sync enabled but no date_field specified. Falling back to full sync.")
        delta_enabled = False

    # set global lowercase rule
    global safe_sql_name
    _orig_safe = safe_sql_name
    def _lower_override(name, coerce_lower=lower):
        return _orig_safe(name, coerce_lower=coerce_lower)
    safe_sql_name = _lower_override

    if engine == 'mssql':
        conn = connect_mssql(server, database, username, password, port)
    else:
        conn = connect_mysql(server, database, username, password, port)

    files = list_allowed_dbfs(folder, include)

    print(f"Found {len(files)} DBFs to load from {folder}")
    if delta_enabled:
        print(f"Delta sync enabled: using field '{date_field}' for incremental updates")

    total_rows = 0
    sync_updates = {}  # Track new sync times per table
    
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        tgt = f"{tpref}{base}{tsuff}"
        table_key = f"{tracking_key}|{tgt}"
        
        try:
            since_date = None
            if delta_enabled:
                # Get last sync time for this table
                last_sync_str = sync_tracking.get(table_key)
                if last_sync_str:
                    try:
                        since_date = datetime.fromisoformat(last_sync_str)
                        print(f"  Delta sync: {tgt} - syncing records since {since_date}")
                    except:
                        print(f"  Warning: Could not parse last sync time for {tgt}, doing full sync")
                        since_date = None
            
            if drop_recreate:
                # Drop & recreate fresh table, then load
                inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=True, date_field=date_field, since_date=since_date)
            else:
                # Reconcile schema; optional truncate then load
                inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False, date_field=date_field, since_date=since_date)
                if trunc:
                    truncate_table(conn, engine, tgt, schema)
                    inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False, date_field=date_field, since_date=since_date)

            total_rows += inserted
            
            # Update sync tracking (always update after successful sync)
            if inserted >= 0:  # Update even if 0 rows (indicates successful sync)
                # Store ISO with local timezone info
                sync_updates[table_key] = datetime.now().astimezone().isoformat()
            
            sync_mode = "delta" if (delta_enabled and since_date) else "full"
            print(f"Loaded {inserted:>8} rows → {tgt} ({batches} batch/es) [{sync_mode} sync]")
        except Exception as e:
            print(f"ERROR {base}: {e}")

    # Save updated sync tracking (always save to track last sync time)
    if sync_updates:
        sync_tracking.update(sync_updates)
        save_sync_tracking(sync_tracking)

    try:
        conn.close()
    except Exception:
        pass
    print(f"ALL DONE. Inserted {total_rows} rows total.")

def cli_init(path: Optional[str] = None):
    print("\n=== Initial Setup (creates a saved config) ===")
    engine = input("Engine (mysql/mssql) [mysql]: ").strip().lower() or 'mysql'
    server = input("RDS server/host: ").strip()
    port = input("Port [3306 for mysql, 1433 for mssql]: ").strip()
    port = int(port) if port else (3306 if engine == 'mysql' else 1433)
    database = input("Database name: ").strip()
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    folder = input(r"Path to ksv\data folder (e.g., C:/ksv/data): ").strip()
    schema = input("Schema (mssql) [dbo]: ").strip() if engine == 'mssql' else None
    schema = schema or ('dbo' if engine == 'mssql' else None)

    cfg = {
        'rds': {
            'engine': engine,
            'server': server,
            'port': port,
            'database': database,
            'username': username,
            'password': password,
            'schema': schema,
        },
        'source': {'folder': folder},
        'load': {
            'drop_recreate': True,
            'truncate_before_load': False,
            'batch_size': 1000,
            'table_prefix': '',
            'table_suffix': '',
            'coerce_lowercase_names': True,
        },
    }
    cfg_path = path or default_config_path()
    save_config(cfg, cfg_path)
    print(f"Saved config to {cfg_path}.")

# ---------- DearPyGui GUI (kept, optional) ----------

def run_gui():
    try:
        import dearpygui.dearpygui as dpg
    except Exception:
        print("DearPyGui not available. Install with: pip install dearpygui")
        sys.exit(1)

    dpg.create_context()
    state = {'files': [], 'folder': ''}

    def log(msg: str):
        old = dpg.get_value("log_box") if dpg.does_item_exist("log_box") else ""
        dpg.set_value("log_box", (old + msg + "\n")[-20000:])

    def scan_dbfs():
        folder = dpg.get_value("folder_input")
        if not folder or not os.path.isdir(folder):
            log("Please select a valid folder.")
            return
        files = list_allowed_dbfs(folder)
        state['folder'] = folder
        state['files'] = files
        dpg.configure_item("files_list", items=[os.path.basename(f) for f in files])
        dpg.set_value("count_text", f"Found: {len(files)}")
        log(f"Scanned {len(files)} DBFs.")

    def choose_folder(sender, app_data):
        path = app_data.get('file_path_name') or app_data.get('current_path')
        if path and os.path.isdir(path):
            dpg.set_value("folder_input", path)
            scan_dbfs()

    def save_cfg():
        engine = dpg.get_value("engine_combo")
        server = dpg.get_value("server_input")
        port = int(dpg.get_value("port_input") or (3306 if engine=='mysql' else 1433))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input") or ''
        schema = dpg.get_value("schema_input") if engine=='mssql' else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ''
        tsuff = dpg.get_value("tsuff_input") or ''

        # Persist current settings (including password) to config
        cfg = {
            'rds': {
                'engine': engine,
                'server': server,
                'port': port,
                'database': database,
                'username': username,
                'password': password,
                'schema': schema,
            },
            'source': {'folder': state.get('folder','')},
            'load': {
                'drop_recreate': recreate,
                'truncate_before_load': trunc,
                'batch_size': batch,
                'table_prefix': tpref,
                'table_suffix': tsuff,
                'coerce_lowercase_names': True,
            }
        }

        path = save_config(cfg, default_config_path())
        log(f"Saved config to {path}.")

    def upload_selected():
        sel_indices = dpg.get_value("files_list") or []
        if not sel_indices:
            log("No DBFs selected.")
            return

        engine = dpg.get_value("engine_combo")
        server = dpg.get_value("server_input")
        port = int(dpg.get_value("port_input") or (3306 if engine=='mysql' else 1433))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input")
        schema = dpg.get_value("schema_input") if engine=='mssql' else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ''
        tsuff = dpg.get_value("tsuff_input") or ''

        # Persist current settings (including password) to config
        cfg = {
            'rds': {
                'engine': engine,
                'server': server,
                'port': port,
                'database': database,
                'username': username,
                'password': password,
                'schema': schema,
            },
            'source': {'folder': state.get('folder','')},
            'load': {
                'drop_recreate': recreate,
                'truncate_before_load': trunc,
                'batch_size': batch,
                'table_prefix': tpref,
                'table_suffix': tsuff,
                'coerce_lowercase_names': True,
            }
        }
        save_config(cfg, default_config_path())

        try:
            if engine == 'mssql':
                conn = connect_mssql(server, database, username, password, port)
            else:
                conn = connect_mysql(server, database, username, password, port)
        except Exception as e:
            log(f"Connection failed: {e}")
            return

        total_rows = 0
        try:
            for idx in sel_indices:
                p = state['files'][idx]
                base = os.path.splitext(os.path.basename(p))[0]
                tgt = f"{tpref}{base}{tsuff}"
                try:
                    if recreate:
                        inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=True)
                    else:
                        inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False)
                        if trunc:
                            truncate_table(conn, engine, tgt, schema)
                            inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False)
                    total_rows += inserted
                    log(f"Loaded {inserted} rows into {tgt} in {batches} batch(es).")
                except Exception as e:
                    log(f"Error loading {base}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        log(f"DONE. Total inserted rows: {total_rows}.")

    # Layout
    with dpg.window(label="VFP DBF → RDS Uploader (DearPyGui)", width=980, height=700, pos=(50, 50)):
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="ksv/data folder", tag="folder_input", width=600)
            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("folder_dialog"))
            dpg.add_button(label="Scan DBFs", callback=scan_dbfs)
            dpg.add_text("Found: 0", tag="count_text")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_listbox(items=[], tag="files_list", width=400, num_items=16, label="DBF Tables")
            with dpg.group():
                dpg.add_combo(["mysql","mssql"], default_value="mysql", label="Engine", tag="engine_combo")
                dpg.add_input_text(label="Server/Host", tag="server_input")
                dpg.add_input_text(label="Port", tag="port_input", default_value="3306")
                dpg.add_input_text(label="Database", tag="db_input")
                dpg.add_input_text(label="Username", tag="user_input")
                dpg.add_input_text(label="Password", tag="pwd_input", password=True)
                dpg.add_input_text(label="Schema (mssql)", tag="schema_input", default_value="dbo")
                dpg.add_checkbox(label="Drop & recreate tables", tag="recreate_chk", default_value=True)
                dpg.add_checkbox(label="Truncate before load", tag="trunc_chk")
                dpg.add_input_text(label="Batch size", tag="batch_input", default_value="1000")
                dpg.add_input_text(label="Table prefix", tag="tpref_input")
                dpg.add_input_text(label="Table suffix", tag="tsuff_input")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Config", callback=save_cfg)
                    dpg.add_button(label="Upload Selected", callback=upload_selected)
        dpg.add_separator()
        dpg.add_input_text(tag="log_box", multiline=True, readonly=True, height=220, width=940)

    with dpg.file_dialog(directory_selector=True, show=False, callback=choose_folder, tag="folder_dialog"):
        dpg.add_file_extension(".dbf")

    dpg.create_viewport(title='DBF → RDS Uploader', width=1100, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# ---------- Tkinter GUI (Modernized) ----------

def run_gui_tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import pathlib

    # Resolve or ask for \ksv\ once; remember it next to the script
    ksv = find_ksv_root()
    if not ksv:
        messagebox.showinfo("Select ksv Folder", "Please select your 'ksv' folder once. We'll remember it.")
        chosen = filedialog.askdirectory(title="Select the 'ksv' folder (e.g., C:\\ksv)")
        if not chosen or os.path.basename(chosen).lower() != "ksv":
            messagebox.showerror("ksv folder", "You must select a folder literally named 'ksv'.")
            return
        hint_path = pathlib.Path(__file__).resolve().parent / "ksv_path.txt"
        hint_path.write_text(chosen, encoding="utf-8")
        ksv = chosen

    state = {"files": [], "folder": ""}

    def log(msg: str):
        log_box.configure(state="normal")
        log_box.insert("end", msg + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def scan_dbfs():
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder", "Please select a valid folder.")
            return
        files = list_allowed_dbfs(folder)
        state["folder"] = folder
        state["files"] = files
        files_list.delete(0, "end")
        for f in files:
            files_list.insert("end", os.path.basename(f))
        count_var.set(f"Found: {len(files)}")
        log(f"Scanned {len(files)} DBFs.")

    def browse_folder():
        folder = filedialog.askdirectory(title="Select ksv/data folder", initialdir=os.path.join(ksv, "data"))
        if folder:
            folder_var.set(folder)
            scan_dbfs()

    def load_cfg():
        try:
            cfg = load_config(default_config_path())
            if 'profiles' in cfg:
                prof = next(iter(cfg['profiles'].keys()))
                eff = cfg['profiles'][prof]
            else:
                eff = cfg
            r = eff['rds']; s = eff['source']; l = eff['load']
            admin = eff.get('admin_api', {})
            engine_var.set(r.get('engine','mysql'))
            server_var.set(r.get('server',''))
            port_var.set(str(r.get('port', 3306 if r.get('engine','mysql')=='mysql' else 1433)))
            db_var.set(r.get('database',''))
            user_var.set(r.get('username',''))
            schema_var.set(r.get('schema','') if r.get('engine','mysql')=='mssql' else '')
            folder_var.set(s.get('folder','') or os.path.join(ksv, "data"))
            trunc_var.set(1 if l.get('truncate_before_load') else 0)
            recreate_var.set(1 if l.get('drop_recreate', True) else 0)
            batch_var.set(str(l.get('batch_size',1000)))
            tpref_var.set(l.get('table_prefix',''))
            tsuff_var.set(l.get('table_suffix',''))
            # Load password if available
            if 'password' in r:
                pwd_var.set(r['password'])
            
            # Load delta sync settings (map None to empty string for UI)
            delta = eff.get('delta_sync', {})
            delta_enabled_var.set(1 if delta.get('enabled', False) else 0)
            _date_field_val = delta.get('date_field')
            delta_date_field_var.set(_date_field_val if isinstance(_date_field_val, str) and _date_field_val else '')
            delta_interval_var.set(str(delta.get('auto_sync_interval_seconds', 3600)))

            # Admin API settings
            admin_enabled_var.set(1 if admin.get('enabled') else 0)
            admin_base_url_var.set(admin.get('base_url', ''))
            admin_api_key_var.set(admin.get('api_key', ''))
            admin_store_id_var.set(str(admin.get('store_id', '')))
            
            log("Loaded values from existing config.")
        except FileNotFoundError:
            messagebox.showinfo("Load Config", "No config found yet. Fill the form and click Save Config.")
        except Exception as e:
            messagebox.showerror("Load Config", str(e))

    def save_cfg():
        engine = engine_var.get()
        try:
            cfg = {
                "rds": {
                    "engine": engine,
                    "server": server_var.get().strip(),
                    "port": int(port_var.get() or (3306 if engine == "mysql" else 1433)),
                    "database": db_var.get().strip(),
                    "username": user_var.get().strip(),
                    "password": pwd_var.get(),
                    "schema": schema_var.get().strip() if engine == "mssql" else None,
                },
                "source": {"folder": folder_var.get().strip() or os.path.join(ksv, "data")},
                "load": {
                    "drop_recreate": bool(recreate_var.get()),
                    "truncate_before_load": bool(trunc_var.get()),
                    "batch_size": int(batch_var.get() or 1000),
                    "table_prefix": tpref_var.get().strip(),
                    "table_suffix": tsuff_var.get().strip(),
                    "coerce_lowercase_names": True,
                },
                "admin_api": {
                    "enabled": bool(admin_enabled_var.get()),
                    "base_url": admin_base_url_var.get().strip(),
                    "api_key": admin_api_key_var.get().strip(),
                    "store_id": admin_store_id_var.get().strip(),
                },
                "delta_sync": {
                    "enabled": bool(delta_enabled_var.get()),
                    "date_field": delta_date_field_var.get().strip() or None,
                    "auto_sync_interval_seconds": int(delta_interval_var.get() or 3600),
                },
            }
        except ValueError:
            messagebox.showerror("Config", "Port, Batch size, and Auto-sync interval must be numbers.")
            return

        path = save_config(cfg, default_config_path())
        log(f"Saved config to {path}.")

    def upload_selected():
        sel = files_list.curselection()
        if not sel:
            messagebox.showinfo("Upload", "No DBFs selected.")
            return

        engine = engine_var.get()
        server = server_var.get().strip()
        database = db_var.get().strip()
        username = user_var.get().strip()
        password = pwd_var.get()
        try:
            port = int(port_var.get() or (3306 if engine == "mysql" else 1433))
        except ValueError:
            messagebox.showerror("Upload", "Port must be a number.")
            return
        schema = schema_var.get().strip() if engine == "mssql" else None

        trunc = bool(trunc_var.get())
        recreate = bool(recreate_var.get())
        try:
            batch = int(batch_var.get() or 1000)
        except ValueError:
            messagebox.showerror("Upload", "Batch size must be a number.")
            return

        tpref = tpref_var.get().strip()
        tsuff = tsuff_var.get().strip()

        # Admin API override
        use_admin = bool(admin_enabled_var.get())
        if use_admin:
            try:
                creds = fetch_admin_creds(
                    admin_base_url_var.get().strip(),
                    admin_api_key_var.get().strip(),
                    admin_store_id_var.get().strip(),
                )
                engine = creds.get('engine', engine)
                server = creds['host']
                port = int(creds.get('port', port))
                database = creds['database']
                username = creds['username']
                password = creds['password']
                schema = creds.get('schema') if engine == 'mssql' else None
                log("Fetched store credentials from Admin backend.")
            except Exception as e:
                messagebox.showerror("Admin creds", f"Failed to fetch creds: {e}")
                return

        # Persist current settings to config
        try:
            cfg_persist = {
                "rds": {
                    "engine": engine,
                    "server": server,
                    "port": port,
                    "database": database,
                    "username": username,
                    "password": password,
                    "schema": schema,
                },
                "source": {"folder": folder_var.get().strip() or os.path.join(ksv, "data")},
                "load": {
                    "drop_recreate": bool(recreate),
                    "truncate_before_load": bool(trunc),
                    "batch_size": int(batch),
                    "table_prefix": tpref,
                    "table_suffix": tsuff,
                    "coerce_lowercase_names": True,
                },
                "admin_api": {
                    "enabled": bool(admin_enabled_var.get()),
                    "base_url": admin_base_url_var.get().strip(),
                    "api_key": admin_api_key_var.get().strip(),
                    "store_id": admin_store_id_var.get().strip(),
                },
            }
            save_config(cfg_persist, default_config_path())
        except Exception:
            pass

        try:
            if engine == 'mssql':
                conn = connect_mssql(server, database, username, password, port)
            else:
                conn = connect_mysql(server, database, username, password, port)
        except Exception as e:
            messagebox.showerror("Connection failed", str(e))
            return

        # Delta sync settings from UI
        delta_enabled = bool(delta_enabled_var.get()) and not recreate
        date_field_cfg = (delta_date_field_var.get() or '').strip()
        date_field = date_field_cfg if delta_enabled and date_field_cfg else None

        # Load sync tracking for incremental sync
        sync_tracking = load_sync_tracking()
        profile_key = 'default'
        folder = folder_var.get().strip() or os.path.join(ksv, "data")
        if use_admin and admin_store_id_var.get().strip():
            tracking_key = f"{profile_key}|{admin_store_id_var.get().strip()}|{database}|{folder}"
        else:
            tracking_key = f"{profile_key}|{database}|{folder}"

        total_rows = 0
        sync_updates = {}
        try:
            for idx in sel:
                p = state["files"][idx]
                base = os.path.splitext(os.path.basename(p))[0]
                tgt = f"{tpref}{base}{tsuff}"
                table_key = f"{tracking_key}|{tgt}"
                try:
                    # Determine since_date for delta sync
                    since_date = None
                    if date_field:
                        last_sync_str = sync_tracking.get(table_key)
                        if last_sync_str:
                            try:
                                since_date = datetime.fromisoformat(last_sync_str)
                                log(f"Delta: {tgt} since {since_date}")
                            except Exception:
                                since_date = None

                    if recreate:
                        inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=True,
                                                        date_field=date_field, since_date=since_date)
                    else:
                        inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False,
                                                        date_field=date_field, since_date=since_date)
                        if trunc:
                            truncate_table(conn, engine, tgt, schema)
                            inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False,
                                                            date_field=date_field, since_date=since_date)
                    total_rows += inserted
                    log(f"Loaded {inserted} rows into {tgt} in {batches} batch(es).")

                    # Update sync tracking on success
                    if inserted >= 0:
                        # Store ISO with local timezone info
                        sync_updates[table_key] = datetime.now().astimezone().isoformat()
                except Exception as e:
                    log(f"Error loading {base}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # Save updated last sync timestamps
        if sync_updates:
            sync_tracking.update(sync_updates)
            save_sync_tracking(sync_tracking)

        log(f"DONE. Total inserted rows: {total_rows}.")

    def run_easy():
        default_data = os.path.join(ksv, "data")
        if not state.get("folder"):
            if os.path.isdir(default_data):
                state["folder"] = default_data
                folder_var.set(default_data)
            else:
                messagebox.showwarning("Folder", "Select your ksv\\data folder first.")
                return
        ffiles = list_allowed_dbfs(state["folder"])
        if not ffiles:
            messagebox.showinfo("Run (Easy)", "No DBF files found in selected folder.")
            return
        files_list.selection_clear(0, "end")
        for i in range(len(ffiles)):
            files_list.selection_set(i)
        recreate_var.set(1)
        trunc_var.set(0)
        upload_selected()

    # Placeholder for update function (defined later)
    update_auto_sync_status = lambda: None
    
    def start_auto_sync_gui():
        """Start auto-sync from GUI."""
        global _auto_sync_thread
        if _auto_sync_thread and _auto_sync_thread.is_alive():
            messagebox.showinfo("Auto-Sync", "Auto-sync is already running.")
            return
        
        # Save config first to ensure settings are current
        save_cfg()
        
        # Start in background
        start_auto_sync_background(default_config_path(), None)
        auto_sync_status_var.set("Running")
        log("Auto-sync started. Check log for progress.")
        if callable(update_auto_sync_status):
            root.after(1000, update_auto_sync_status)

    def stop_auto_sync_gui():
        """Stop auto-sync from GUI."""
        global _auto_sync_thread
        if _auto_sync_thread and _auto_sync_thread.is_alive():
            stop_auto_sync()
            auto_sync_status_var.set("Stopped")
            log("Auto-sync stopped.")
        else:
            messagebox.showinfo("Auto-Sync", "Auto-sync is not running.")


    def edit_config_yaml():
        """Open config YAML file in an editor window."""
        cfg_path = default_config_path()
        
        if not os.path.exists(cfg_path):
            if not messagebox.askyesno("Create Config", "Config file doesn't exist. Create it?"):
                return
            save_cfg()  # Create default config
        
        # Create editor window (responsive to screen size)
        editor = tk.Toplevel(root)
        editor.title("Edit Config YAML")
        
        # Get screen dimensions for editor
        editor_screen_width = editor.winfo_screenwidth()
        editor_screen_height = editor.winfo_screenheight()
        
        # Responsive editor size
        editor_width = max(700, min(int(editor_screen_width * 0.6), 1200))
        editor_height = max(500, min(int(editor_screen_height * 0.7), 900))
        editor_x = (editor_screen_width - editor_width) // 2
        editor_y = (editor_screen_height - editor_height) // 2
        
        editor.geometry(f"{editor_width}x{editor_height}+{editor_x}+{editor_y}")
        editor.minsize(600, 400)
        editor.configure(bg=bg_color)
        
        tk.Label(editor, text=f"Editing: {cfg_path}", font=label_font, bg=bg_color, fg=text_color).pack(pady=5)
        
        text_frame = tk.Frame(editor, bg=entry_bg, relief="flat", bd=1)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        yaml_text = tk.Text(text_frame, wrap="word", font=("Consolas", 9), bg="#fafafa", fg=text_color)
        yaml_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=yaml_text.yview)
        yaml_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        
        # Load current config
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                yaml_text.insert("1.0", f.read())
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config: {e}")
            editor.destroy()
            return
        
        def save_yaml():
            try:
                content = yaml_text.get("1.0", "end-1c")
                # Validate YAML
                yaml.safe_load(content)
                # Save
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", "Config saved successfully!")
                log("Config YAML saved. Reload config to apply changes.")
                editor.destroy()
                load_cfg()  # Reload config into UI
            except yaml.YAMLError as e:
                messagebox.showerror("YAML Error", f"Invalid YAML:\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save config: {e}")
        
        btn_frame = tk.Frame(editor, bg=bg_color)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=save_yaml, style='Accent.TButton').pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=editor.destroy, style='Modern.TButton').pack(side="left", padx=5)

    # --- Modern UI Setup ---
    root = tk.Tk()
    root.title("VFP DBF → RDS Uploader")
    
    # Get screen dimensions for responsive design
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate window size - adapt to screen but ensure usability
    # Use more of the screen on smaller displays, less on larger ones
    if screen_height < 900:
        # Small screens (laptops) - use 90% to maximize space
        window_width = max(950, min(int(screen_width * 0.9), screen_width - 40))
        window_height = max(650, min(int(screen_height * 0.9), screen_height - 40))
    elif screen_height < 1200:
        # Medium screens - use 85%
        window_width = max(1000, min(int(screen_width * 0.85), 1500))
        window_height = max(700, min(int(screen_height * 0.85), 1000))
    else:
        # Large screens - use 75% (more reasonable)
        window_width = max(1000, min(int(screen_width * 0.75), 1600))
        window_height = max(700, min(int(screen_height * 0.75), 1000))
    
    # Center window on screen
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.minsize(950, 650)  # Minimum window size - ensures all sections visible
    
    # Modern color scheme
    bg_color = "#f5f5f5"
    header_color = "#2c3e50"
    accent_color = "#3498db"
    accent_hover = "#2980b9"
    success_color = "#27ae60"
    text_color = "#2c3e50"
    entry_bg = "#ffffff"
    
    root.configure(bg=bg_color)
    
    # Responsive font sizing based on screen resolution
    # Base font size scales with screen height
    base_font_size = max(8, int(screen_height / 100))
    title_font_size = max(12, int(screen_height / 60))
    
    title_font = ("Segoe UI", title_font_size, "bold")
    label_font = ("Segoe UI", base_font_size)
    button_font = ("Segoe UI", base_font_size)
    header_font = ("Segoe UI", base_font_size + 1, "bold")
    
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Style configurations
    style.configure('Title.TLabel', font=title_font, background=bg_color, foreground=header_color)
    style.configure('Header.TLabel', font=("Segoe UI", 10, "bold"), background=bg_color, foreground=text_color)
    style.configure('Modern.TButton', font=button_font, padding=8)
    style.configure('Accent.TButton', font=button_font, padding=8)
    style.map('Accent.TButton',
              background=[('active', accent_hover), ('!active', accent_color)],
              foreground=[('active', 'white'), ('!active', 'white')])
    
    # Use grid layout for better control over section proportions
    root.grid_rowconfigure(2, weight=1)  # Middle section expands
    root.grid_rowconfigure(3, weight=0)  # Log section fixed minimum
    root.grid_columnconfigure(0, weight=1)
    
    # Title frame (compact, fixed height)
    title_height = max(40, min(int(window_height * 0.06), 60))
    title_frame = tk.Frame(root, bg=header_color, height=title_height)
    title_frame.grid(row=0, column=0, sticky="ew")
    title_frame.grid_propagate(False)
    title_frame.grid_columnconfigure(0, weight=1)
    title_label = tk.Label(title_frame, text="VFP DBF → RDS Uploader", 
                          font=title_font, bg=header_color, fg="white")
    title_label.grid(row=0, column=0)
    title_frame.grid_rowconfigure(0, weight=1)

    # Top: folder row with modern styling (compact, fixed height)
    pad_x = max(8, int(window_width / 100))
    pad_y = max(6, int(window_height / 120))
    top = tk.Frame(root, bg=bg_color, padx=pad_x, pady=pad_y)
    top.grid(row=1, column=0, sticky="ew")
    top.grid_columnconfigure(1, weight=1)

    default_folder = os.path.join(ksv, "data") if os.path.isdir(os.path.join(ksv, "data")) else ksv
    folder_var = tk.StringVar(value=default_folder)

    # Responsive folder entry width
    folder_entry_width = max(35, int(window_width / 18))
    tk.Label(top, text="Data Folder", font=label_font, bg=bg_color, fg=text_color).grid(row=0, column=0, padx=(0, 6))
    folder_entry = tk.Entry(top, textvariable=folder_var, width=folder_entry_width, font=label_font, 
                           bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
                           highlightbackground="#ddd", highlightcolor=accent_color)
    folder_entry.grid(row=0, column=1, padx=4, sticky="ew")
    
    ttk.Button(top, text="Browse", command=browse_folder, style='Modern.TButton').grid(row=0, column=2, padx=4)
    ttk.Button(top, text="Scan DBFs", command=scan_dbfs, style='Accent.TButton').grid(row=0, column=3, padx=4)
    count_var = tk.StringVar(value="Found: 0")
    tk.Label(top, textvariable=count_var, font=label_font, bg=bg_color, fg=text_color).grid(row=0, column=4, padx=(8, 0))

    # Middle: list + connection form (takes most space, but leaves room for log)
    mid = tk.Frame(root, bg=bg_color, padx=pad_x, pady=max(4, pad_y // 2))
    mid.grid(row=2, column=0, sticky="nsew")
    # Ensure the only row in 'mid' expands to fill available vertical space
    mid.grid_rowconfigure(0, weight=1)
    mid.grid_columnconfigure(0, weight=3)  # Tables list gets 60% width
    mid.grid_columnconfigure(1, weight=2)  # Settings gets 40% width

    # Left: Listbox with modern styling (responsive to available space)
    left_frame = tk.Frame(mid, bg=bg_color)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, pad_x // 2))
    left_frame.grid_rowconfigure(1, weight=1)  # List container expands
    left_frame.grid_columnconfigure(0, weight=1)
    
    tk.Label(left_frame, text="DBF Tables", font=header_font, 
            bg=bg_color, fg=text_color).grid(row=0, column=0, sticky="w", pady=(0, 3))
    
    list_container = tk.Frame(left_frame, bg=entry_bg, relief="flat", bd=1)
    list_container.grid(row=1, column=0, sticky="nsew")
    list_container.grid_rowconfigure(0, weight=1)
    list_container.grid_columnconfigure(0, weight=1)
    
    # Listbox fills available vertical space (no fixed height limit)
    # Calculate reasonable minimum rows based on screen
    min_listbox_rows = max(10, min(int(window_height / 50), 20))
    files_list = tk.Listbox(list_container, selectmode="extended", height=min_listbox_rows,
                           font=label_font, bg=entry_bg, fg=text_color,
                           selectbackground=accent_color, selectforeground="white",
                           relief="flat", bd=0, highlightthickness=0)
    files_list.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
    
    yscroll = ttk.Scrollbar(list_container, orient="vertical", command=files_list.yview)
    yscroll.grid(row=0, column=1, sticky="ns")
    files_list.configure(yscrollcommand=yscroll.set)

    # Update last sync label when selection changes
    def _format_local_time(iso_str: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_str)
            # If timezone-aware, convert to local; else assume local
            if dt.tzinfo is not None:
                dt = dt.astimezone()
            return dt.strftime("%m/%d/%Y %I:%M %p")
        except Exception:
            return iso_str

    def update_last_sync_label(event=None):
        try:
            sel = files_list.curselection()
            if not sel or len(sel) != 1:
                last_sync_var.set("—")
                return
            idx = sel[0]
            p = state.get("files", [])[idx]
            if not p:
                last_sync_var.set("—")
                return
            base = os.path.splitext(os.path.basename(p))[0]
            tpref = tpref_var.get().strip()
            tsuff = tsuff_var.get().strip()
            tgt = f"{tpref}{base}{tsuff}"
            profile_key = 'default'
            folder = folder_var.get().strip() or os.path.join(ksv, "data")
            database = db_var.get().strip()
            table_key = f"{profile_key}|{database}|{folder}|{tgt}"
            tracking = load_sync_tracking()
            last = tracking.get(table_key)
            last_sync_var.set(_format_local_time(last) if last else "Never")
        except Exception:
            last_sync_var.set("—")
    files_list.bind('<<ListboxSelect>>', update_last_sync_label)

    # Right: Form with modern card-like appearance (scrollable for overflow)
    # Create outer frame for scrollable content
    right_outer = tk.Frame(mid, bg=bg_color)
    right_outer.grid(row=0, column=1, sticky="nsew", padx=(pad_x // 2, 0))
    right_outer.grid_rowconfigure(0, weight=1)
    right_outer.grid_columnconfigure(0, weight=1)
    
    # Create canvas for scrolling
    right_canvas = tk.Canvas(right_outer, bg=entry_bg, highlightthickness=0)
    right_scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
    right_scrollable_frame = tk.Frame(right_canvas, bg=entry_bg)
    
    # Create the window for the scrollable frame
    canvas_window = right_canvas.create_window((0, 0), window=right_scrollable_frame, anchor="nw")
    
    def update_scroll_region(event=None):
        """Update scroll region when frame content changes."""
        right_canvas.configure(scrollregion=right_canvas.bbox("all"))
    
    def update_canvas_width(event=None):
        """Update scrollable frame width to match canvas width."""
        canvas_width = event.width if event else right_canvas.winfo_width()
        if canvas_width > 1:
            right_canvas.itemconfig(canvas_window, width=canvas_width)
            update_scroll_region()
    
    # Bind events for scrolling
    right_scrollable_frame.bind("<Configure>", update_scroll_region)
    right_canvas.bind('<Configure>', update_canvas_width)
    right_canvas.configure(yscrollcommand=right_scrollbar.set)
    
    right_canvas.grid(row=0, column=0, sticky="nsew")
    right_scrollbar.grid(row=0, column=1, sticky="ns")
    
    # Bind mousewheel to canvas (Windows/Mac)
    def _on_mousewheel(event):
        # Only scroll if mouse is over the canvas
        if right_canvas.winfo_containing(event.x_root, event.y_root):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Also support Linux mousewheel (Button-4/Button-5)
    def _on_linux_scroll(event):
        if event.num == 4:
            right_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            right_canvas.yview_scroll(1, "units")
    right_canvas.bind_all("<Button-4>", _on_linux_scroll)
    right_canvas.bind_all("<Button-5>", _on_linux_scroll)
    
    # Use right_scrollable_frame instead of right_frame for all widgets
    right_frame = right_scrollable_frame
    right_frame.configure(relief="flat", bd=1, padx=max(8, pad_x), pady=max(6, pad_y // 2))
    right_frame.grid_columnconfigure(1, weight=1)

    tk.Label(right_frame, text="Connection Settings", font=header_font, 
            bg=entry_bg, fg=header_color).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    engine_var = tk.StringVar(value="mysql")
    tk.Label(right_frame, text="Engine", font=label_font, bg=entry_bg, fg=text_color).grid(row=1, column=0, sticky="w", pady=4)
    engine_menu = ttk.OptionMenu(right_frame, engine_var, "mysql", "mysql", "mssql")
    engine_menu.grid(row=1, column=1, sticky="we", padx=4, pady=4)

    server_var = tk.StringVar()
    port_var = tk.StringVar(value="3306")
    db_var = tk.StringVar()
    user_var = tk.StringVar()
    pwd_var = tk.StringVar()
    schema_var = tk.StringVar()
    trunc_var = tk.IntVar(value=0)
    recreate_var = tk.IntVar(value=1)
    batch_var = tk.StringVar(value="1000")
    tpref_var = tk.StringVar()
    tsuff_var = tk.StringVar()
    
    # Delta sync variables
    delta_enabled_var = tk.IntVar(value=0)
    delta_date_field_var = tk.StringVar()
    delta_interval_var = tk.StringVar(value="3600")
    auto_sync_status_var = tk.StringVar(value="Stopped")
    last_sync_var = tk.StringVar(value="Never")

    # Admin API variables
    admin_enabled_var = tk.IntVar(value=0)
    admin_base_url_var = tk.StringVar()
    admin_api_key_var = tk.StringVar()
    admin_store_id_var = tk.StringVar()

    # Responsive entry field width
    # Entry width based on window width (settings panel is ~40% of window)
    entry_width = max(20, int(window_width * 0.35 / 12))
    row = 2
    fields = [
        ("Server/Host", server_var, entry_width),
        ("Port", port_var, max(8, entry_width // 2)),
        ("Database", db_var, entry_width),
        ("Username", user_var, entry_width),
        ("Password", pwd_var, entry_width, True),  # Password field
    ]
    
    row_pad = 3  # Fixed smaller padding
    manual_db_widgets = []
    for label, var, width, *args in fields:
        is_password = len(args) > 0 and args[0]
        tk.Label(right_frame, text=label, font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
        entry = tk.Entry(right_frame, textvariable=var, width=width, font=label_font,
                        bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
                        highlightbackground="#ddd", highlightcolor=accent_color)
        if is_password:
            entry.config(show="*")
        entry.grid(row=row, column=1, sticky="we", padx=3, pady=row_pad)
        manual_db_widgets.append(entry)
        row += 1

    tk.Label(right_frame, text="Schema (mssql)", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    schema_entry = tk.Entry(right_frame, textvariable=schema_var, width=entry_width, font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color)
    schema_entry.grid(row=row, column=1, sticky="we", padx=3, pady=row_pad)
    manual_db_widgets.append(engine_menu)
    manual_db_widgets.append(schema_entry)
    row += 1

    tk.Checkbutton(right_frame, text="Drop & recreate tables (safe)", variable=recreate_var,
                  font=label_font, bg=entry_bg, fg=text_color, selectcolor=entry_bg,
                  activebackground=entry_bg, activeforeground=text_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=row_pad)
    row += 1
    tk.Checkbutton(right_frame, text="Truncate before load", variable=trunc_var,
                  font=label_font, bg=entry_bg, fg=text_color, selectcolor=entry_bg,
                  activebackground=entry_bg, activeforeground=text_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=row_pad)
    row += 1

    # Admin Backend (toggle to fetch creds from admin)
    tk.Label(right_frame, text="Admin Backend", font=header_font, 
            bg=entry_bg, fg=header_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 3))
    row += 1

    tk.Checkbutton(right_frame, text="Use Admin Backend creds", variable=admin_enabled_var,
                  font=label_font, bg=entry_bg, fg=text_color, selectcolor=entry_bg,
                  activebackground=entry_bg, activeforeground=text_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=row_pad)
    row += 1

    tk.Label(right_frame, text="Base URL", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=admin_base_url_var, width=entry_width, font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="we", padx=3, pady=row_pad)
    row += 1

    tk.Label(right_frame, text="API key", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=admin_api_key_var, width=entry_width, font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="we", padx=3, pady=row_pad)
    row += 1

    tk.Label(right_frame, text="Store ID", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=admin_store_id_var, width=max(12, entry_width // 2), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1

    tk.Label(right_frame, text="Batch size", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=batch_var, width=max(8, entry_width // 2), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1

    tk.Label(right_frame, text="Table prefix", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=tpref_var, width=max(15, int(entry_width * 0.7)), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1

    tk.Label(right_frame, text="Table suffix", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tsuff_entry = tk.Entry(right_frame, textvariable=tsuff_var, width=max(15, int(entry_width * 0.7)), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color)
    tsuff_entry.grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1

    # Delta sync section
    tk.Label(right_frame, text="Delta Sync", font=header_font, 
            bg=entry_bg, fg=header_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 3))
    row += 1
    
    tk.Checkbutton(right_frame, text="Enable delta sync", variable=delta_enabled_var,
                  font=label_font, bg=entry_bg, fg=text_color, selectcolor=entry_bg,
                  activebackground=entry_bg, activeforeground=text_color).grid(row=row, column=0, columnspan=2, sticky="w", pady=row_pad)
    row += 1
    
    tk.Label(right_frame, text="Date field", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    date_entry = tk.Entry(right_frame, textvariable=delta_date_field_var, width=max(15, int(entry_width * 0.7)), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color)
    date_entry.grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1
    
    tk.Label(right_frame, text="Auto-sync interval (sec)", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Entry(right_frame, textvariable=delta_interval_var, width=max(15, int(entry_width * 0.7)), font=label_font,
            bg=entry_bg, relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#ddd", highlightcolor=accent_color).grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1

    # Show last sync time for selected table
    tk.Label(right_frame, text="Last sync", font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=0, sticky="w", pady=row_pad)
    tk.Label(right_frame, textvariable=last_sync_var, font=label_font, bg=entry_bg, fg=text_color).grid(row=row, column=1, sticky="w", padx=3, pady=row_pad)
    row += 1
    
    # Auto-sync status and controls
    auto_sync_frame = tk.Frame(right_frame, bg=entry_bg)
    auto_sync_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(4, 0))
    
    tk.Label(auto_sync_frame, text="Status:", font=label_font, bg=entry_bg, fg=text_color).pack(side="left", padx=(0, 5))
    status_label = tk.Label(auto_sync_frame, textvariable=auto_sync_status_var, font=label_font, 
                           bg=entry_bg, fg=text_color)
    status_label.pack(side="left", padx=(0, 10))
    
    # Make status_label accessible to update function
    def update_auto_sync_status_with_label():
        """Periodically update auto-sync status with label reference."""
        global _auto_sync_thread
        if _auto_sync_thread and _auto_sync_thread.is_alive():
            auto_sync_status_var.set("Running")
            status_label.config(fg=success_color)
            root.after(2000, update_auto_sync_status_with_label)
        else:
            auto_sync_status_var.set("Stopped")
            status_label.config(fg=text_color)
    
    # Replace the update function
    update_auto_sync_status = update_auto_sync_status_with_label
    
    ttk.Button(auto_sync_frame, text="Start", command=start_auto_sync_gui, style='Modern.TButton', width=8).pack(side="left", padx=2)
    ttk.Button(auto_sync_frame, text="Stop", command=stop_auto_sync_gui, style='Modern.TButton', width=8).pack(side="left", padx=2)
    row += 1

    # Button frame with modern styling (reduced padding)
    btn_pad = 4
    btn_frame = tk.Frame(right_frame, bg=entry_bg)
    btn_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(btn_pad, 0))
    
    ttk.Button(btn_frame, text="Load Config", command=load_cfg, style='Modern.TButton').pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(btn_frame, text="Save Config", command=save_cfg, style='Modern.TButton').pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(btn_frame, text="Edit YAML", command=edit_config_yaml, style='Modern.TButton').pack(side="left", padx=2, fill="x", expand=True)
    
    action_frame = tk.Frame(right_frame, bg=entry_bg)
    action_frame.grid(row=row+1, column=0, columnspan=2, sticky="we", pady=(btn_pad, 0))
    
    ttk.Button(action_frame, text="Upload Selected", command=upload_selected, style='Accent.TButton').pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(action_frame, text="Run (Easy)", command=run_easy, style='Accent.TButton').pack(side="left", padx=2, fill="x", expand=True)

    right_frame.grid_columnconfigure(1, weight=1)

    # Disable/enable manual DB fields based on admin toggle
    def _apply_admin_toggle(*_):
        use_admin = bool(admin_enabled_var.get())
        state = "disabled" if use_admin else "normal"
        # Engine menu is a ttk OptionMenu which doesn’t support state directly; skip it
        for w in manual_db_widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass
    admin_enabled_var.trace_add('write', _apply_admin_toggle)
    # Apply initial
    _apply_admin_toggle()

    # Log box with modern styling (guaranteed minimum height, always visible)
    log_frame = tk.Frame(root, bg=bg_color, padx=pad_x, pady=max(4, pad_y // 2))
    log_frame.grid(row=3, column=0, sticky="ew")
    log_frame.grid_columnconfigure(0, weight=1)
    log_frame.grid_rowconfigure(1, weight=1)  # Log container can grow
    
    # Calculate log height as percentage of window (ensures visibility)
    # Minimum 15% of window height, maximum 25%
    log_min_px = max(80, int(window_height * 0.15))  # Always at least 15% of window
    log_max_px = min(int(window_height * 0.25), 200)  # Max 25% or 200px
    log_height_px = max(log_min_px, min(log_max_px, int(window_height * 0.2)))
    
    tk.Label(log_frame, text="Activity Log", font=header_font, 
            bg=bg_color, fg=text_color).grid(row=0, column=0, sticky="w", pady=(0, 3))
    
    log_container = tk.Frame(log_frame, bg=entry_bg, relief="flat", bd=1)
    log_container.grid(row=1, column=0, sticky="ew")
    log_container.config(height=log_height_px)
    log_container.grid_propagate(False)  # Maintain minimum height
    log_container.grid_rowconfigure(0, weight=1)
    log_container.grid_columnconfigure(0, weight=1)
    
    log_font_size = max(8, base_font_size - 1)
    # Calculate rows based on pixel height (approx 20px per row)
    log_rows = max(8, int(log_height_px / 22))
    log_box = tk.Text(log_container, height=log_rows, wrap="word", 
                     font=("Consolas", log_font_size),
                     bg="#fafafa", fg=text_color, relief="flat", bd=0,
                     padx=max(6, pad_x // 2), pady=max(4, pad_y // 2))
    log_box.grid(row=0, column=0, sticky="nsew")
    log_box.configure(state="disabled")
    
    log_scroll = ttk.Scrollbar(log_container, orient="vertical", command=log_box.yview)
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_box.configure(yscrollcommand=log_scroll.set)

    # Auto-load config on launch if it exists
    if os.path.exists(default_config_path()):
        try:
            load_cfg()
            log("Config loaded automatically on launch.")
        except Exception as e:
            log(f"Could not auto-load config: {e}")
    
    # Auto-scan default folder on open
    scan_dbfs()
    # Initialize last sync label based on current selection
    try:
        update_last_sync_label()
    except Exception:
        pass
    
    # Start status update timer
    update_auto_sync_status()
    
    root.mainloop()

# ---------- Auto-sync functionality ----------

_auto_sync_thread = None
_auto_sync_stop = threading.Event()

def run_auto_sync(cfg_path: Optional[str] = None, profile: Optional[str] = None):
    """Run sync in a loop based on interval in config."""
    raw_cfg = load_config(cfg_path) if cfg_path else load_config()
    cfg = resolve_profile(raw_cfg, profile)
    
    delta_cfg = cfg.get('delta_sync', {})
    interval_seconds = int(delta_cfg.get('auto_sync_interval_seconds', 3600))  # Default 1 hour
    
    print(f"Starting auto-sync (interval: {interval_seconds}s, profile: {profile or 'default'})")
    print("Press Ctrl+C to stop")
    
    while not _auto_sync_stop.is_set():
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running sync...")
            run_headless(cfg_path, profile, auto_sync=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sync complete. Next sync in {interval_seconds}s")
        except KeyboardInterrupt:
            print("\nStopping auto-sync...")
            break
        except Exception as e:
            print(f"Error during auto-sync: {e}")
            print(f"Retrying in {interval_seconds}s...")
        
        # Wait for interval, but check stop event periodically
        for _ in range(interval_seconds):
            if _auto_sync_stop.wait(timeout=1):
                break
    
    print("Auto-sync stopped.")

def start_auto_sync_background(cfg_path: Optional[str] = None, profile: Optional[str] = None):
    """Start auto-sync in a background thread."""
    global _auto_sync_thread
    if _auto_sync_thread and _auto_sync_thread.is_alive():
        print("Auto-sync already running.")
        return
    
    _auto_sync_stop.clear()
    _auto_sync_thread = threading.Thread(target=run_auto_sync, args=(cfg_path, profile), daemon=True)
    _auto_sync_thread.start()
    print("Auto-sync started in background.")

def stop_auto_sync():
    """Stop the running auto-sync."""
    global _auto_sync_thread
    _auto_sync_stop.set()
    if _auto_sync_thread:
        _auto_sync_thread.join(timeout=5)
    print("Auto-sync stopped.")

# ---------- Main ----------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="VFP DBF → RDS Uploader")
    ap.add_argument('--config', help='Path to YAML config (defaults to ksv\\vfp_uploader.yaml if found, else AppData)')
    ap.add_argument('--init', action='store_true', help='Run interactive setup wizard and save config')
    ap.add_argument('--gui', action='store_true', help='Launch Tkinter GUI')
    ap.add_argument('--dpg', action='store_true', help='Launch DearPyGui GUI')
    ap.add_argument('--profile', help='Profile name in config (when using profiles)')
    ap.add_argument('--auto-sync', action='store_true', help='Run auto-sync (periodic sync based on config interval)')
    ap.add_argument('--stop-sync', action='store_true', help='Stop running auto-sync')
    args = ap.parse_args()

    if args.stop_sync:
        stop_auto_sync()
        return

    if args.init:
        cli_init(args.config)
        run_headless(args.config, profile=args.profile)
        return

    if args.auto_sync:
        try:
            run_auto_sync(args.config, profile=args.profile)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        return

    if args.dpg:
        run_gui()
        return

    if args.gui:
        run_gui_tk()
        return

    cfg_path = args.config if args.config else (default_config_path() if os.path.exists(default_config_path()) else None)
    debug_config_path("Resolved config", cfg_path or "<none>")

    if cfg_path:
        run_headless(cfg_path, profile=args.profile)
        return

    # No config yet → show Tk GUI for first-run simplicity
    run_gui_tk()
    return

if __name__ == '__main__':
    main()
