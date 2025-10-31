#!/usr/bin/env python3
r"""
VFP DBF → RDS Uploader
- Lists .DBF tables from a chosen ksv\data folder
- Infers schema from DBF and auto-creates tables in RDS (SQL Server or MySQL)
- Bulk inserts data with batching
- Supports GUI mode (Tkinter or DearPyGui) and headless mode via --config YAML
- Stores config next to \ksv\ by default (grandma-proof)
- Supports multi-store profiles and secure per-store passwords via keyring
- Fresh-load safety: drop_recreate option to avoid duplicates

Prereqs (Windows):
  pip install dbfread pyyaml keyring pyodbc mysql-connector-python dearpygui

Example config.yaml (single or multi-profile):
---
# Single-profile format (no 'profiles' key):
rds:
  engine: mssql
  server: 34.207.129.23
  port: 1433
  database: "Spirits Demo GC"
  username: ASIDEMOGC
  password: __KEYRING__
  schema: dbo
source:
  folder: "C:/ksv/data"
  include: ["inv.dbf", "stk.dbf"]   # omit for all DBFs
load:
  drop_recreate: true          # guarantees no duplicates
  truncate_before_load: false  # ignored if drop_recreate is true
  batch_size: 1000
  table_prefix: ""
  table_suffix: ""
  coerce_lowercase_names: true
  nvarchar_default: 255

# Multi-profile format:
profiles:
  store6885:
    rds:
      engine: mssql
      server: 34.207.129.23
      port: 1433
      database: SpiritsDemo_6885
      username: store6885_ingest
      password: __KEYRING__
      schema: dbo
    source:
      folder: "C:/ksv/data_6885"
    load:
      drop_recreate: true
      truncate_before_load: false
      batch_size: 1000
      table_prefix: ""
      table_suffix: "_6885"
      coerce_lowercase_names: true
"""

import os
import re
import sys
import glob
import yaml
import getpass
from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict, Any, Iterable, Tuple, Optional

# ---------------- Config persistence helpers ----------------
APP_NAME = "StoreInsights"
CONFIG_NAME = "vfp_uploader.yaml"
KEYRING_SERVICE = "VFP_RDS_Uploader"

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

def _keyring_key(server: str, database: str, username: str) -> str:
    return f"{server}|{database}|{username}"

def put_password(server: str, database: str, username: str, password: str):
    if keyring:
        keyring.set_password(KEYRING_SERVICE, _keyring_key(server, database, username), password)

def get_password(server: str, database: str, username: str) -> Optional[str]:
    if keyring:
        return keyring.get_password(KEYRING_SERVICE, _keyring_key(server, database, username))
    return None

def ensure_password(cfg: dict) -> str:
    r = cfg.get('rds', {})
    pw = r.get('password')
    if pw and pw != "__KEYRING__":
        # Honor explicit password in config; optionally seed keyring
        try:
            put_password(r['server'], r['database'], r['username'], pw)
        except Exception:
            pass
        return pw
    if pw == "__KEYRING__":
        got = get_password(r['server'], r['database'], r['username'])
        if got:
            return got
        raise RuntimeError("Password is __KEYRING__ but no secret found in keyring. Run --init or enter in GUI.")
    got = get_password(r.get('server',''), r.get('database',''), r.get('username',''))
    if got:
        return got
    raise RuntimeError("No RDS password available. Run --init or enter in GUI.")

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


# Optional secure password storage
try:
    import keyring  # Windows Credential Manager / macOS Keychain
except Exception:
    keyring = None

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

def map_dbf_field_to_sql(field, engine: str = "mssql", nvarchar_default: int = 255) -> str:
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

    # mysql
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

def iter_dbf_rows(dbf_path: str) -> Tuple[List[str], Iterable[List[Any]]]:
    table = open_dbf(dbf_path)
    field_names = [f.name for f in table.fields]
    def gen():
        for rec in table:
            yield [coerce_value(rec.get(name)) for name in field_names]
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

def bulk_insert(conn, engine: str, table: str, dbf_path: str, batch_size: int = 1000, schema: str = None, recreate: bool = False) -> Tuple[int, int]:
    dbf_obj = open_dbf(dbf_path)
    ensure_table(conn, engine, table, dbf_obj.fields, schema, recreate=recreate)

    cols_existing = existing_columns(conn, engine, table, schema)
    col_names, row_iter = iter_dbf_rows(dbf_path)
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

def run_headless(cfg_path: Optional[str] = None, profile: Optional[str] = None):
    raw_cfg = load_config(cfg_path) if cfg_path else load_config()
    cfg = resolve_profile(raw_cfg, profile)

    engine = cfg['rds'].get('engine', 'mssql')
    server = cfg['rds']['server']
    port = int(cfg['rds'].get('port', 1433 if engine=='mssql' else 3306))
    database = cfg['rds']['database']
    username = cfg['rds']['username']
    password = ensure_password(cfg)
    schema = cfg['rds'].get('schema') if engine=='mssql' else None

    folder = cfg['source']['folder']
    include = cfg['source'].get('include')

    drop_recreate = bool(cfg['load'].get('drop_recreate', False))
    trunc = bool(cfg['load'].get('truncate_before_load', False))
    batch = int(cfg['load'].get('batch_size', 1000))
    tpref = cfg['load'].get('table_prefix', '')
    tsuff = cfg['load'].get('table_suffix', '')
    lower = bool(cfg['load'].get('coerce_lowercase_names', True))

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

    total_rows = 0
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        tgt = f"{tpref}{base}{tsuff}"
        try:
            if drop_recreate:
                # Drop & recreate fresh table, then load
                inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=True)
            else:
                # Reconcile schema; optional truncate then load
                inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False)
                if trunc:
                    truncate_table(conn, engine, tgt, schema)
                    inserted, batches = bulk_insert(conn, engine, tgt, p, batch, schema, recreate=False)

            total_rows += inserted
            print(f"Loaded {inserted:>8} rows → {tgt} ({batches} batch/es)")
        except Exception as e:
            print(f"ERROR {base}: {e}")

    try:
        conn.close()
    except Exception:
        pass
    print(f"ALL DONE. Inserted {total_rows} rows total.")

def cli_init(path: Optional[str] = None):
    print("\n=== Initial Setup (creates a saved config + stores password in keyring) ===")
    engine = input("Engine (mssql/mysql) [mssql]: ").strip().lower() or 'mssql'
    server = input("RDS server/host: ").strip()
    port = input("Port [1433 for mssql, 3306 for mysql]: ").strip()
    port = int(port) if port else (1433 if engine == 'mssql' else 3306)
    database = input("Database name: ").strip()
    username = input("Username: ").strip()
    password = getpass.getpass("Password (stored securely in system keyring): ")
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
            'drop_recreate': True,        # default safer
            'truncate_before_load': False,
            'batch_size': 1000,
            'table_prefix': '',
            'table_suffix': '',
            'coerce_lowercase_names': True,
        },
    }
    cfg_path = path or default_config_path()
    put_password(server, database, username, password)
    save_config(cfg, cfg_path)
    print(f"Saved config to {cfg_path} and password to system keyring.")

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
        port = int(dpg.get_value("port_input") or (1433 if engine=='mssql' else 3306))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input") or ''
        schema = dpg.get_value("schema_input") if engine=='mssql' else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ''
        tsuff = dpg.get_value("tsuff_input") or ''

        # Persist current settings (including password) to config for automated runs
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

        # Optionally seed keyring, but keep plaintext in YAML
        if password:
            try:
                put_password(cfg['rds']['server'], cfg['rds']['database'], cfg['rds']['username'], password)
            except Exception as e:
                log(f"Warning: could not store password in keyring: {e}")

        path = save_config(cfg, default_config_path())
        log(f"Saved config to {path} (password stored in system keyring).")

    def upload_selected():
        sel_indices = dpg.get_value("files_list") or []
        if not sel_indices:
            log("No DBFs selected.")
            return

        engine = dpg.get_value("engine_combo")
        server = dpg.get_value("server_input")
        port = int(dpg.get_value("port_input") or (1433 if engine=='mssql' else 3306))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input")
        schema = dpg.get_value("schema_input") if engine=='mssql' else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ''
        tsuff = dpg.get_value("tsuff_input") or ''

        if password:
            try:
                put_password(server, database, username, password)
            except Exception as e:
                log(f"Warning: could not store password in keyring: {e}")

        # Persist current settings (including password) to config for automated runs
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
                dpg.add_combo(["mssql","mysql"], default_value="mssql", label="Engine", tag="engine_combo")
                dpg.add_input_text(label="Server/Host", tag="server_input")
                dpg.add_input_text(label="Port", tag="port_input", default_value="1433")
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

# ---------- Tkinter GUI (recommended) ----------

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
            engine_var.set(r.get('engine','mssql'))
            server_var.set(r.get('server',''))
            port_var.set(str(r.get('port', 1433 if r.get('engine','mssql')=='mssql' else 3306)))
            db_var.set(r.get('database',''))
            user_var.set(r.get('username',''))
            schema_var.set(r.get('schema','dbo') if r.get('engine','mssql')=='mssql' else '')
            folder_var.set(s.get('folder','') or os.path.join(ksv, "data"))
            trunc_var.set(1 if l.get('truncate_before_load') else 0)
            recreate_var.set(1 if l.get('drop_recreate', True) else 0)
            batch_var.set(str(l.get('batch_size',1000)))
            tpref_var.set(l.get('table_prefix',''))
            tsuff_var.set(l.get('table_suffix',''))
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
                    "port": int(port_var.get() or (1433 if engine == "mssql" else 3306)),
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
            }
        except ValueError:
            messagebox.showerror("Config", "Port and Batch size must be numbers.")
            return

        pw = pwd_var.get()
        if pw:
            try:
                put_password(cfg["rds"]["server"], cfg["rds"]["database"], cfg["rds"]["username"], pw)
            except Exception as e:
                log(f"Warning: could not store password in keyring: {e}")

        # Save right next to \ksv\
        path = save_config(cfg, default_config_path())
        log(f"Saved config to {path} (password stored in system keyring).")

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
            port = int(port_var.get() or (1433 if engine == "mssql" else 3306))
        except ValueError:
            messagebox.showerror("Upload", "Port must be a number.")
            return
        schema = schema_var.get().strip() if engine == "mssql" else None

        if password:
            try:
                put_password(server, database, username, password)
            except Exception as e:
                log(f"Warning: could not store password in keyring: {e}")

        # Persist current settings (including password) to config for automated runs
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
            }
            save_config(cfg_persist, default_config_path())
        except Exception:
            # Non-fatal if persisting fails
            pass

        try:
            if engine == 'mssql':
                conn = connect_mssql(server, database, username, password, port)
            else:
                conn = connect_mysql(server, database, username, password, port)
        except Exception as e:
            messagebox.showerror("Connection failed", str(e))
            return

        trunc = bool(trunc_var.get())
        recreate = bool(recreate_var.get())
        try:
            batch = int(batch_var.get() or 1000)
        except ValueError:
            messagebox.showerror("Upload", "Batch size must be a number.")
            try:
                conn.close()
            except Exception:
                pass
            return

        tpref = tpref_var.get().strip()
        tsuff = tsuff_var.get().strip()

        total_rows = 0
        try:
            for idx in sel:
                p = state["files"][idx]
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

    def run_easy():
        # Select all DBFs in ksv\data; force drop&recreate
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

    # --- UI ---
    root = tk.Tk()
    root.title("VFP DBF → RDS Uploader (Tkinter)")
    root.geometry("1100x780")

    # Top: folder row
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")

    default_folder = os.path.join(ksv, "data") if os.path.isdir(os.path.join(ksv, "data")) else ksv
    folder_var = tk.StringVar(value=default_folder)

    ttk.Label(top, text="ksv/data folder").pack(side="left")
    ttk.Entry(top, textvariable=folder_var, width=80).pack(side="left", padx=8)
    ttk.Button(top, text="Browse", command=browse_folder).pack(side="left", padx=4)
    ttk.Button(top, text="Scan DBFs", command=scan_dbfs).pack(side="left", padx=4)
    count_var = tk.StringVar(value="Found: 0")
    ttk.Label(top, textvariable=count_var).pack(side="left", padx=8)

    # Middle: list + connection form
    mid = ttk.Frame(root, padding=8)
    mid.pack(fill="both", expand=True)

    # Listbox
    left = ttk.Frame(mid)
    left.pack(side="left", fill="both", expand=True, padx=(0, 6))
    ttk.Label(left, text="DBF Tables").pack(anchor="w")
    files_list = tk.Listbox(left, selectmode="extended", height=20)
    files_list.pack(fill="both", expand=True)
    yscroll = ttk.Scrollbar(left, orient="vertical", command=files_list.yview)
    files_list.configure(yscrollcommand=yscroll.set)
    yscroll.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

    # Form
    right = ttk.Frame(mid)
    right.pack(side="left", fill="y")

    engine_var = tk.StringVar(value="mssql")
    ttk.Label(right, text="Engine").grid(row=0, column=0, sticky="w")
    ttk.OptionMenu(right, engine_var, "mssql", "mssql", "mysql").grid(row=0, column=1, sticky="we", padx=4, pady=2)

    server_var = tk.StringVar()
    port_var = tk.StringVar(value="1433")
    db_var = tk.StringVar()
    user_var = tk.StringVar()
    pwd_var = tk.StringVar()
    schema_var = tk.StringVar(value="dbo")
    trunc_var = tk.IntVar(value=0)
    recreate_var = tk.IntVar(value=1)  # default on (safer)
    batch_var = tk.StringVar(value="1000")
    tpref_var = tk.StringVar()
    tsuff_var = tk.StringVar()

    row = 1
    for label, var, width in [
        ("Server/Host", server_var, 30),
        ("Port", port_var, 10),
        ("Database", db_var, 30),
        ("Username", user_var, 30),
    ]:
        ttk.Label(right, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(right, textvariable=var, width=width).grid(row=row, column=1, sticky="we", padx=4, pady=2)
        row += 1

    ttk.Label(right, text="Password").grid(row=row, column=0, sticky="w")
    ttk.Entry(right, textvariable=pwd_var, show="*", width=30).grid(row=row, column=1, sticky="we", padx=4, pady=2)
    row += 1

    ttk.Label(right, text="Schema (mssql)").grid(row=row, column=0, sticky="w")
    ttk.Entry(right, textvariable=schema_var, width=30).grid(row=row, column=1, sticky="we", padx=4, pady=2)
    row += 1

    ttk.Checkbutton(right, text="Drop & recreate tables (safe)", variable=recreate_var).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    ttk.Checkbutton(right, text="Truncate before load (only if not recreating)", variable=trunc_var).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    ttk.Label(right, text="Batch size").grid(row=row, column=0, sticky="w")
    ttk.Entry(right, textvariable=batch_var, width=12).grid(row=row, column=1, sticky="w", padx=4, pady=2)
    row += 1

    ttk.Label(right, text="Table prefix").grid(row=row, column=0, sticky="w")
    ttk.Entry(right, textvariable=tpref_var, width=18).grid(row=row, column=1, sticky="w", padx=4, pady=2)
    row += 1

    ttk.Label(right, text="Table suffix").grid(row=row, column=0, sticky="w")
    ttk.Entry(right, textvariable=tsuff_var, width=18).grid(row=row, column=1, sticky="w", padx=4, pady=2)
    row += 1

    btns = ttk.Frame(right)
    btns.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
    ttk.Button(btns, text="Load Config", command=load_cfg).pack(side="left", padx=4)
    ttk.Button(btns, text="Save Config", command=save_cfg).pack(side="left", padx=4)
    ttk.Button(btns, text="Upload Selected", command=upload_selected).pack(side="left", padx=4)
    ttk.Button(btns, text="Run (Easy)", command=run_easy).pack(side="left", padx=4)
    row += 1

    # Log box
    log_frame = ttk.Frame(root, padding=8)
    log_frame.pack(fill="both", expand=False)
    ttk.Label(log_frame, text="Log").pack(anchor="w")
    log_box = tk.Text(log_frame, height=12, wrap="word")
    log_box.pack(fill="both", expand=True)
    log_box.configure(state="disabled")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_box.yview)
    log_box.configure(yscrollcommand=log_scroll.set)
    log_scroll.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

    for c in range(2):
        right.grid_columnconfigure(c, weight=1)

    # Auto-scan default folder on open
    scan_dbfs()
    root.mainloop()

# ---------- Main ----------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="VFP DBF → RDS Uploader")
    ap.add_argument('--config', help='Path to YAML config (defaults to ksv\\vfp_uploader.yaml if found, else AppData)')
    ap.add_argument('--init', action='store_true', help='Run interactive setup wizard and save config + password')
    ap.add_argument('--gui', action='store_true', help='Launch Tkinter GUI')
    ap.add_argument('--dpg', action='store_true', help='Launch DearPyGui GUI')
    ap.add_argument('--profile', help='Profile name in config (when using profiles)')
    args = ap.parse_args()

    if args.init:
        cli_init(args.config)
        run_headless(args.config, profile=args.profile)
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
