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
  Optional for system tray: pip install pystray pillow pywin32

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
  truncate_before_load: false  # NOTE: When true, delta sync is automatically disabled during truncate operation
  batch_size: 1000
  table_prefix: ""
  table_suffix: ""
  coerce_lowercase_names: true
  nvarchar_default: 255
  date_range_filter:
    enabled: false              # Enable date range filtering for initial loads (useful for limiting historical data)
    date_field: "date"          # Field name to use for filtering (defaults to delta_sync.date_field if not specified)
    start_date: "2021-01-01"    # Start date (YYYY-MM-DD), omit for no lower bound
    end_date: "2024-12-31"      # End date (YYYY-MM-DD), omit for no upper bound
delta_sync:
  enabled: true                # Enable incremental sync
  date_field: "date"            # Default field name in DBF to use for date filtering (case-insensitive)
                                # Common fields: "date", "updated", "created", "timestamp", "cdate"
  date_fields_per_table:       # Optional: Override date_field per table (table name without .dbf extension)
    inv: "cdate"                # Example: inv table uses "cdate" for new items created
    # stk: "updated"            # Example: stk table uses "updated"
    # poh: "tstamp"             # Example: poh table uses "tstamp"
  related_table_date_fields:   # Optional: Use date field from related table for filtering (when table has no date field)
    jnl:                        # Example: jnl table has no tstamp, so use jnh.tstamp
      related_table: "jnh"       # Related table name (without .dbf extension)
      join_field_local: "sale"  # Field in current table to match on
      join_field_related: "sale" # Field in related table to match on
      date_field_related: "tstamp" # Date field in related table to check
    sll:                        # Example: sll table has no tstamp, so use slh.tstamp
      related_table: "slh"       # Related table name (without .dbf extension)
      join_field_local: "listnum"  # Field in current table to match on
      join_field_related: "listnum" # Field in related table to match on
      date_field_related: "tstamp" # Date field in related table to check
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
      date_range_filter:
        enabled: true
        date_field: "date"
        start_date: "2022-01-01"  # Only load data from 2022 onwards
        end_date: ""              # No upper bound
    delta_sync:
      enabled: true
      date_field: "date"
      auto_sync_interval_seconds: 1800  # 30 minutes

Usage:
  python vfp_dbf_to_rdsv2.py                          # Launch GUI (default)
  python vfp_dbf_to_rdsv2.py --gui                    # Launch GUI (explicit)
  python vfp_dbf_to_rdsv2.py --headless --config path/to/config.yaml  # Run once without GUI
  python vfp_dbf_to_rdsv2.py --headless --silent --config path/to/config.yaml  # Run silently (Task Scheduler)
  python vfp_dbf_to_rdsv2.py --auto-sync              # Run auto-sync (periodic, headless, for Task Scheduler)
  python vfp_dbf_to_rdsv2.py --auto-sync --silent      # Run auto-sync silently (Task Scheduler)
  python vfp_dbf_to_rdsv2.py --profile store6885      # Use specific profile
"""

import logging
import os
import re
import sys
import json
import shutil
from pathlib import Path
from urllib import request as urlrequest
from urllib import parse as urlparse
import yaml
import getpass
import time
import threading
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Iterable, Tuple, Optional, Union

try:
    from platformdirs import user_config_path
except ImportError:  # pragma: no cover - optional dependency for legacy environments
    user_config_path = None  # type: ignore[assignment]

# ---------------- Config persistence helpers ----------------
APP_NAME = "StoreInsights"
CONFIG_NAME = "vfp_uploader.yaml"
SYNC_TRACKING_NAME = "vfp_sync_tracking.yaml"

LOGGER_NAME = "store_insights.uploader"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: Optional[int] = None) -> None:
    """Configure application-wide logging."""
    if level is None:
        level_name = os.getenv("STORE_INSIGHTS_LOG_LEVEL", "INFO").upper()
        resolved_level = logging.getLevelName(level_name)
        level = resolved_level if isinstance(resolved_level, int) else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger(LOGGER_NAME).setLevel(level)


logger = logging.getLogger(LOGGER_NAME)


def debug_config_path(msg: str, path: Union[str, Path]) -> None:
    logger.debug("%s: %s", msg, path)


def find_ksv_root(start: Optional[Union[str, Path]] = None) -> Optional[str]:
    r"""
    Try to locate a base folder for ksv/data-style structures.
    Priority:
      1) explicit start path or its parents
      2) current working dir or its parents
      3) script dir or its parents
      4) hint file 'ksv_path.txt' next to this script

    Historically this required the folder name to literally be 'ksv'. For
    backwards compatibility we still prefer that, but fall back to whatever
    path the user selected if no such folder is found.
    """
    start_path: Optional[Path] = None
    if start:
        resolved = Path(start).resolve()
        if resolved.is_file():
            resolved = resolved.parent
        if resolved.exists():
            start_path = resolved

    candidates: List[Path] = []
    if start_path:
        candidates.append(start_path)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)

    for base in candidates:
        p = base
        for _ in range(6):
            if p.name.lower() == "ksv":
                return str(p)
            if p.parent == p:
                break
            p = p.parent

    if start_path:
        return str(start_path)

    hint = Path(__file__).resolve().parent / "ksv_path.txt"
    if hint.exists():
        txt = hint.read_text(encoding="utf-8").strip()
        candidate = Path(txt) if txt else None
        if candidate and candidate.is_dir():
            return str(candidate.resolve())
    return None


def ksv_config_path(ksv_root: Union[str, Path]) -> str:
    return str(Path(ksv_root) / CONFIG_NAME)


def _app_config_root() -> Path:
    if user_config_path:
        return Path(user_config_path(APP_NAME))
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / ".config" / APP_NAME


def default_config_path() -> str:
    # Prefer local \ksv\vfp_uploader.yaml
    ksv = find_ksv_root()
    if ksv:
        return ksv_config_path(ksv)
    base = _app_config_root()
    base.mkdir(parents=True, exist_ok=True)
    return str(base / CONFIG_NAME)


def save_config(cfg: dict, path: Optional[Union[str, Path]] = None) -> str:
    resolved_path = Path(path) if path else Path(default_config_path())
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    debug_config_path("Saved config", resolved_path)
    return str(resolved_path)


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    resolved_path = Path(path) if path else Path(default_config_path())
    debug_config_path("Loading config", resolved_path)
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_sync_tracking_path() -> str:
    """Get path to sync tracking file (stores last sync timestamps per table/store)."""
    ksv = find_ksv_root()
    if ksv:
        return str(Path(ksv) / SYNC_TRACKING_NAME)
    base = _app_config_root()
    base.mkdir(parents=True, exist_ok=True)
    return str(base / SYNC_TRACKING_NAME)


def load_sync_tracking(path: Optional[Union[str, Path]] = None) -> dict:
    """Load sync tracking data (last sync timestamps per table)."""
    resolved_path = Path(path) if path else Path(default_sync_tracking_path())
    if resolved_path.exists():
        try:
            with resolved_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception:
            return {}
    return {}


def save_sync_tracking(tracking: dict, path: Optional[Union[str, Path]] = None) -> str:
    """Save sync tracking data."""
    resolved_path = Path(path) if path else Path(default_sync_tracking_path())
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(tracking, f, sort_keys=False)
    debug_config_path("Saved sync tracking", resolved_path)
    return str(resolved_path)


def resolve_profile(cfg: dict, profile: Optional[str]) -> dict:
    """
    Return a single-profile dict with keys rds/source/load.
    If 'profiles' exists, require --profile and return that entry.
    Otherwise treat cfg as single-profile.
    """
    if "profiles" in cfg:
        if not profile:
            raise RuntimeError(
                "Config contains multiple profiles. Use --profile <name>."
            )
        if profile not in cfg["profiles"]:
            raise RuntimeError(f"Profile '{profile}' not found in config.")
        return cfg["profiles"][profile]
    return cfg


def ensure_password(cfg: dict) -> str:
    """Read password directly from YAML config."""
    r = cfg.get("rds", {})
    pw = r.get("password")
    if not pw:
        raise RuntimeError(
            "No password specified in config. Please add 'password' to rds section in YAML."
        )
    return pw


# ---------- Admin backend creds fetch ----------
def fetch_admin_creds(base_url: str, api_key: str, store_id: str) -> dict:
    """Fetch per-store DB credentials from admin backend.

    Returns a dict with keys: engine, host, port, database, username, password, schema
    """
    if not base_url:
        raise RuntimeError("Admin base_url is required when admin_api.enabled is true")
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    url = f"{base_url}/uploader/creds?{urlparse.urlencode({'store_id': store_id})}"
    req = urlrequest.Request(url)
    req.add_header("X-API-Key", api_key or "")
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Admin creds fetch failed: HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
            # Basic shape check
            for k in ("host", "port", "database", "username", "password"):
                if k not in data:
                    raise RuntimeError(f"Admin creds missing field: {k}")
            return data
    except Exception as e:
        raise RuntimeError(f"Admin creds fetch error: {e}")


# --- Only scan these DBFs ---
ALLOWED_BASES = {
    "cnt",
    "cat",
    "cus",
    "emp",
    "glb",
    "inv",
    "jnh",
    "jnl",
    "pod",
    "poh",
    "prc",
    "slh",
    "sll",
    "stk",
    "str",
    "upc",
    "vnd",
    "timeclock",
    "hst",
}


def is_allowed_dbf(path: str) -> bool:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    return base in ALLOWED_BASES


def get_sync_directory(source_folder: str) -> str:
    """Get the sync directory path (vfptordssync inside ksv folder)."""
    ksv_root = find_ksv_root(source_folder)
    if ksv_root:
        sync_dir = Path(ksv_root) / "vfptordssync"
    else:
        sync_dir = Path(source_folder).resolve().parent / "vfptordssync"

    sync_dir.mkdir(parents=True, exist_ok=True)
    return str(sync_dir)


def copy_dbf_and_related_files(source_file: str, sync_dir: str) -> str:
    """Copy DBF file and its related CDX and FPT files to sync directory.

    Returns:
        Path to the copied DBF file in sync directory.
    """

    source_path = Path(source_file)
    source_dir = source_path.parent
    base_name = source_path.name
    base_name_no_ext = source_path.stem

    dest_dbf = Path(sync_dir) / base_name
    dest_cdx = Path(sync_dir) / f"{base_name_no_ext}.cdx"
    dest_fpt = Path(sync_dir) / f"{base_name_no_ext}.fpt"

    source_cdx = source_dir / f"{base_name_no_ext}.cdx"
    source_fpt = source_dir / f"{base_name_no_ext}.fpt"

    # Copy DBF file
    try:
        shutil.copy2(source_file, dest_dbf)
    except Exception as e:
        log_to_gui(f"WARNING: Could not copy {base_name}: {e}")
        return source_file  # Fallback to original

    # Copy CDX file if it exists
    if source_cdx.exists():
        try:
            shutil.copy2(source_cdx, dest_cdx)
        except Exception as e:
            log_to_gui(f"WARNING: Could not copy {base_name_no_ext}.cdx: {e}")

    # Copy FPT file if it exists
    if source_fpt.exists():
        try:
            shutil.copy2(source_fpt, dest_fpt)
        except Exception as e:
            log_to_gui(f"WARNING: Could not copy {base_name_no_ext}.fpt: {e}")

    return str(dest_dbf)


def sync_files_to_directory(
    source_folder: str, include: Optional[List[str]] = None
) -> Tuple[str, List[str]]:
    """Copy allowed DBF files and their related files to sync directory.

    Returns:
        Tuple of (sync_directory_path, list_of_copied_dbf_paths)
    """
    sync_dir = get_sync_directory(source_folder)

    # Get list of source DBF files
    source_root = Path(source_folder)
    if include:
        source_files = [str(source_root / f) for f in include]
    else:
        source_files = [str(p) for p in source_root.glob("*.dbf")]

    # Filter to allowed DBFs
    allowed_source_files = [p for p in source_files if is_allowed_dbf(p)]

    # Copy files to sync directory
    copied_files = []
    for source_file in allowed_source_files:
        copied_file = copy_dbf_and_related_files(source_file, sync_dir)
        copied_files.append(copied_file)

    return sync_dir, sorted(copied_files)


def list_allowed_dbfs(folder: str, include: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of *.dbf in folder limited to ALLOWED_BASES.
    If include list is provided, still filter to ALLOWED_BASES.

    Note: This function returns paths to original files.
    Use sync_files_to_directory() to copy files to sync directory first.
    """
    folder_path = Path(folder)
    if include:
        files = [folder_path / f for f in include]
    else:
        files = folder_path.glob("*.dbf")
    return sorted([str(p) for p in files if is_allowed_dbf(str(p))])


from dbfread import DBF, FieldParser


# Safe field parser that handles parsing errors gracefully
class SafeFieldParser(FieldParser):
    """Field parser that returns None instead of raising exceptions on parse errors."""

    def parseD(self, f, d):
        try:
            return super().parseD(f, d)
        except Exception:
            return None

    def parseN(self, f, d):
        try:
            return super().parseN(f, d)
        except Exception:
            return None

    def parseF(self, f, d):
        try:
            return super().parseF(f, d)
        except Exception:
            return None

    def parseL(self, f, d):
        try:
            return super().parseL(f, d)
        except Exception:
            return None

    def parseT(self, f, d):
        try:
            return super().parseT(f, d)
        except Exception:
            return None

    def parseI(self, f, d):
        try:
            return super().parseI(f, d)
        except Exception:
            return None

    def parseB(self, f, d):
        try:
            return super().parseB(f, d)
        except Exception:
            return None

    def parseM(
        self, f, d
    ):  # Memo fields - return empty string to avoid FPT file access
        return ""

    def parseG(self, f, d):  # General/OLE
        return ""

    def parseO(self, f, d):  # Object
        return ""

    def parseP(self, f, d):  # Picture
        return ""


def open_dbf(
    path: str,
    encodings=("latin-1", "cp1252", "cp437", "utf-8"),
    use_safe_parser: bool = True,
) -> DBF:
    """Open DBF file with optional safe parser that handles parsing errors gracefully."""
    last_err = None
    for enc in encodings:
        try:
            if use_safe_parser:
                return DBF(
                    path,
                    encoding=enc,
                    load=False,
                    parserclass=SafeFieldParser,
                    ignore_missing_memofile=True,
                )
            else:
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

    # Pre-import MySQL plugins to ensure they're registered (critical for PyInstaller builds)
    try:
        import mysql.connector.plugins.mysql_native_password
    except ImportError:
        pass
    try:
        import mysql.connector.plugins.caching_sha2_password
    except ImportError:
        pass
    try:
        import mysql.connector.plugins.sha256_password
    except ImportError:
        pass
    try:
        import mysql.connector.plugins.mysql_clear_password
    except ImportError:
        pass
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


def map_dbf_field_to_sql(
    field, engine: str = "mysql", nvarchar_default: int = 255
) -> str:
    ftype = field.type  # 'C', 'N', 'F', 'D', 'T', 'L', 'M', 'I', 'B'
    length = (
        getattr(field, "length", None)
        or getattr(field, "size", None)
        or nvarchar_default
    )
    decimals = getattr(field, "decimal_count", 0)

    if engine == "mssql":
        if ftype == "C":
            return f"NVARCHAR({length if length and length <= 4000 else 'MAX'})"
        if ftype in ("N", "F"):
            if decimals and decimals > 0:
                precision = max(length or 18, decimals + 1)
                scale = decimals
                precision = min(38, precision)
                return f"DECIMAL({precision},{scale})"
            else:
                return (
                    "INT"
                    if (length or 10) <= 9
                    else ("BIGINT" if (length or 19) <= 18 else "DECIMAL(38,0)")
                )
        if ftype == "I":
            return "INT"
        if ftype == "B":
            return "FLOAT"
        if ftype == "D":
            return "DATE"
        if ftype in ("T", "@"):
            return "DATETIME2(3)"
        if ftype == "L":
            return "BIT"
        if ftype == "M":
            return "NVARCHAR(MAX)"
        return f"NVARCHAR({nvarchar_default})"

    # mysql (default)
    if ftype == "C":
        return f"VARCHAR({min(length or nvarchar_default, 65535)})"
    if ftype in ("N", "F"):
        if decimals and decimals > 0:
            precision = max(length or 18, decimals + 1)
            scale = decimals
            return f"DECIMAL({precision},{scale})"
        else:
            return (
                "INT"
                if (length or 10) <= 9
                else ("BIGINT" if (length or 19) <= 18 else "DECIMAL(38,0)")
            )
    if ftype == "I":
        return "INT"
    if ftype == "B":
        return "DOUBLE"
    if ftype == "D":
        return "DATE"
    if ftype in ("T", "@"):
        return "DATETIME"
    if ftype == "L":
        return "TINYINT(1)"
    if ftype == "M":
        return "LONGTEXT"
    return f"VARCHAR({nvarchar_default})"


# ---------- DDL generation & schema reconciliation ----------


def build_create_table_sql(table: str, fields, engine: str, schema: str = None) -> str:
    col_defs = []
    for f in fields:
        col_name = safe_sql_name(f.name)
        col_type = map_dbf_field_to_sql(f, engine)
        col_defs.append(
            f"[{col_name}] {col_type}"
            if engine == "mssql"
            else f"`{col_name}` {col_type}"
        )

    if engine == "mssql":
        target = (
            f"[{schema}].[{safe_sql_name(table)}]"
            if schema
            else f"[{safe_sql_name(table)}]"
        )
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


def connect_mssql(
    server: str, database: str, username: str, password: str, port: int = 1433
):
    if pyodbc is None:
        raise RuntimeError("pyodbc not installed. pip install pyodbc")
    driver = "ODBC Driver 17 for SQL Server"
    conn_str = f"DRIVER={{{{}}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};"
    # Put driver name inside braces with value:
    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};"
    conn = pyodbc.connect(conn_str, autocommit=False)
    try:
        conn.fast_executemany = True
    except Exception:
        pass
    return conn


def connect_mysql(
    host: str, database: str, username: str, password: str, port: int = 3306
):
    if mysql is None:
        raise RuntimeError(
            "mysql-connector-python not installed. pip install mysql-connector-python"
        )
    # Use pure Python implementation to avoid C extension issues in PyInstaller builds
    # This ensures plugins work correctly in frozen executables
    return mysql.connector.connect(
        host=host,
        user=username,
        password=password,
        database=database,
        port=port,
        use_pure=True,  # Force pure Python implementation
    )


# ---------- Helpers ----------


def table_exists(conn, engine: str, table: str, schema: str = None) -> bool:
    cur = conn.cursor()
    if engine == "mssql":
        if schema:
            cur.execute(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """,
                (schema, safe_sql_name(table)),
            )
        else:
            cur.execute(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = ?
            """,
                (safe_sql_name(table),),
            )
        row = cur.fetchone()
        return bool(row)
    else:
        cur.execute("SHOW TABLES LIKE %s", (safe_sql_name(table),))
        return bool(cur.fetchone())


def existing_columns(conn, engine: str, table: str, schema: str = None) -> List[str]:
    cur = conn.cursor()
    if engine == "mssql":
        if schema:
            cur.execute(
                """
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """,
                (schema, safe_sql_name(table)),
            )
        else:
            cur.execute(
                """
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
            """,
                (safe_sql_name(table),),
            )
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
    no_lead_us = wanted[1:] if wanted.startswith("_") else wanted
    if no_lead_us.lower() in existing_lower:
        return existing_lower[no_lead_us.lower()]
    return wanted


def add_missing_column(
    conn, engine: str, table: str, col_name: str, field, schema: str = None
):
    col_type = map_dbf_field_to_sql(field, engine)
    if engine == "mssql":
        target = (
            f"[{schema}].[{safe_sql_name(table)}]"
            if schema
            else f"[{safe_sql_name(table)}]"
        )
        sql = f"ALTER TABLE {target} ADD [{col_name}] {col_type}"
    else:
        target = f"`{safe_sql_name(table)}`"
        sql = f"ALTER TABLE {target} ADD `{col_name}` {col_type}"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def drop_table(conn, engine: str, table: str, schema: str = None):
    cur = conn.cursor()
    if engine == "mssql":
        target = (
            f"[{schema}].[{safe_sql_name(table)}]"
            if schema
            else f"[{safe_sql_name(table)}]"
        )
        cur.execute(f"IF OBJECT_ID(N'{target}', N'U') IS NOT NULL DROP TABLE {target}")
    else:
        target = f"`{safe_sql_name(table)}`"
        cur.execute(f"DROP TABLE IF EXISTS {target}")
    conn.commit()


def create_table_indexes(conn, engine: str, table: str, schema: str = None):
    """
    Create optimized indexes for specific tables.
    - jnh/jnl: Performance indexes for hourly sales queries
    - Other tables: Indexes for duplicate checking performance
    Called automatically when tables are created or recreated.
    """
    table_lower = table.lower()

    cur = conn.cursor()
    indexes = []  # Initialize empty list - will be populated for specific tables

    # Get existing columns to match actual column names
    existing_cols = existing_columns(conn, engine, table, schema)

    try:
        # Normalize column names using safe_sql_name to match table's actual column names
        if table_lower == "jnh":
            col_tstamp = safe_sql_name("tstamp")
            col_sale = safe_sql_name("sale")
        elif table_lower == "jnl":
            col_sale = safe_sql_name("sale")
            col_line = safe_sql_name("line")
            col_rflag = safe_sql_name("rflag")
            col_sku = safe_sql_name("sku")
        elif table_lower in ("inv", "stk", "prc"):
            col_sku = safe_sql_name("sku")
        elif table_lower == "upc":
            col_upc = safe_sql_name("upc")
            col_sku = safe_sql_name("sku")
        elif table_lower == "hst":
            col_sku = safe_sql_name("sku")
            col_date = safe_sql_name("date")
        elif table_lower in ("slh", "sll"):
            col_listnum = safe_sql_name("listnum")
        elif table_lower in ("poh", "pod"):
            col_order = safe_sql_name("order")
            if table_lower == "pod":
                col_sku = safe_sql_name("sku")
                col_store = safe_sql_name("store") if "store" in existing_cols else None
        elif table_lower == "glb":
            col_date = safe_sql_name("date")
        elif table_lower == "cnt":
            col_code = safe_sql_name("code")
        elif table_lower == "cus":
            col_customer = safe_sql_name("customer")
        elif table_lower == "vnd":
            col_vendor = safe_sql_name("vendor")
            col_vcode = safe_sql_name("vcode")
        elif table_lower == "str":
            col_store = safe_sql_name("store")
        elif table_lower == "cat":
            col_cat = safe_sql_name("cat")
        elif table_lower == "emp":
            col_id = safe_sql_name("id")
            col_uid = safe_sql_name("uid")
        else:
            # Table not in our list - no indexes needed
            return

        if engine == "mssql":
            target = (
                f"[{schema}].[{safe_sql_name(table)}]"
                if schema
                else f"[{safe_sql_name(table)}]"
            )

            if table_lower == "jnh":
                # Indexes for jnh table (performance + duplicate checking)
                indexes = [
                    (
                        "idx_jnh_tstamp_sale",
                        f"CREATE INDEX idx_jnh_tstamp_sale ON {target} ([{col_tstamp}], [{col_sale}])",
                    ),
                    (
                        "idx_jnh_tstamp",
                        f"CREATE INDEX idx_jnh_tstamp ON {target} ([{col_tstamp}])",
                    ),
                    (
                        "idx_jnh_sale",
                        f"CREATE INDEX idx_jnh_sale ON {target} ([{col_sale}])",
                    ),
                ]
            elif table_lower == "jnl":
                # Indexes for jnl table (performance + duplicate checking)
                col_date = (
                    choose_column_name("date", existing_cols)
                    or choose_column_name("tdate", existing_cols)
                    or safe_sql_name("date")
                )
                indexes = [
                    (
                        "idx_jnl_sale_line_rflag",
                        f"CREATE INDEX idx_jnl_sale_line_rflag ON {target} ([{col_sale}], [{col_line}], [{col_rflag}])",
                    ),
                    (
                        "idx_jnl_sale_rflag_sku",
                        f"CREATE INDEX idx_jnl_sale_rflag_sku ON {target} ([{col_sale}], [{col_rflag}], [{col_sku}])",
                    ),
                    (
                        "idx_jnl_sale",
                        f"CREATE INDEX idx_jnl_sale ON {target} ([{col_sale}])",
                    ),
                    (
                        "idx_jnl_date_rflag_sku",
                        f"CREATE INDEX idx_jnl_date_rflag_sku ON {target} ([{col_date}], [{col_rflag}], [{col_sku}])",
                    ),
                    (
                        "idx_jnl_sku_rflag_date",
                        f"CREATE INDEX idx_jnl_sku_rflag_date ON {target} ([{col_sku}], [{col_rflag}], [{col_date}])",
                    ),
                    (
                        "idx_jnl_line_rflag_sale",
                        f"CREATE INDEX idx_jnl_line_rflag_sale ON {target} ([{col_line}], [{col_rflag}], [{col_sale}])",
                    ),
                ]
            elif table_lower in ("inv", "stk", "prc"):
                # Index for duplicate checking (SKU)
                indexes = [
                    (
                        f"idx_{table_lower}_sku",
                        f"CREATE INDEX idx_{table_lower}_sku ON {target} ([{col_sku}])",
                    ),
                ]
            elif table_lower == "upc":
                # Composite index for duplicate checking (UPC + SKU)
                indexes = [
                    (
                        "idx_upc_upc_sku",
                        f"CREATE INDEX idx_upc_upc_sku ON {target} ([{col_upc}], [{col_sku}])",
                    ),
                ]
            elif table_lower == "hst":
                # Indexes for hst table (sales history)
                indexes = [
                    (
                        "idx_hst_sku_date",
                        f"CREATE INDEX idx_hst_sku_date ON {target} ([{col_sku}], [{col_date}])",
                    ),
                    (
                        "idx_hst_date_sku",
                        f"CREATE INDEX idx_hst_date_sku ON {target} ([{col_date}], [{col_sku}])",
                    ),
                    (
                        "idx_hst_date",
                        f"CREATE INDEX idx_hst_date ON {target} ([{col_date}])",
                    ),
                ]
            elif table_lower in ("slh", "sll"):
                # Index for duplicate checking (listnum)
                indexes = [
                    (
                        f"idx_{table_lower}_listnum",
                        f"CREATE INDEX idx_{table_lower}_listnum ON {target} ([{col_listnum}])",
                    ),
                ]
            elif table_lower == "poh":
                # Indexes for poh table: duplicate checking + performance for date/status/vendor queries
                # Find actual column names from existing columns
                col_rcvdate = (
                    choose_column_name("rcvdate", existing_cols)
                    or choose_column_name("received_date", existing_cols)
                    or choose_column_name("rcv_date", existing_cols)
                )
                col_status = choose_column_name(
                    "status", existing_cols
                ) or choose_column_name("stat", existing_cols)
                col_vendor = choose_column_name(
                    "vendor", existing_cols
                ) or choose_column_name("vcode", existing_cols)

                # Only create indexes if columns exist
                index_list = [
                    ("idx_poh_order", col_order),
                ]
                if col_rcvdate and col_status and col_vendor:
                    index_list.append(
                        (
                            "idx_poh_status_rcvdate_vendor",
                            f"{col_status}, {col_rcvdate}, {col_vendor}",
                        )
                    )
                if col_rcvdate:
                    index_list.append(("idx_poh_rcvdate", col_rcvdate))
                if col_vendor:
                    index_list.append(("idx_poh_vendor", col_vendor))

                indexes = [
                    (name, f"CREATE INDEX {name} ON {target} ([{cols}])")
                    for name, cols in index_list
                ]
            elif table_lower == "pod":
                # Indexes for pod table (purchase order details)
                index_list = [
                    ("idx_pod_order", col_order),
                    ("idx_pod_sku", col_sku),
                ]
                if col_store:
                    index_list.append(("idx_pod_store", col_store))
                # Composite index for common query pattern: sku + order
                index_list.append(("idx_pod_sku_order", f"{col_sku}, {col_order}"))

                indexes = [
                    (name, f"CREATE INDEX {name} ON {target} ([{cols}])")
                    for name, cols in index_list
                ]
            elif table_lower == "glb":
                # Index for duplicate checking (date)
                indexes = [
                    (
                        "idx_glb_date",
                        f"CREATE INDEX idx_glb_date ON {target} ([{col_date}])",
                    ),
                ]
            elif table_lower == "cnt":
                # Index for duplicate checking (code)
                indexes = [
                    (
                        "idx_cnt_code",
                        f"CREATE INDEX idx_cnt_code ON {target} ([{col_code}])",
                    ),
                ]
            elif table_lower == "cus":
                # Index for duplicate checking (customer)
                indexes = [
                    (
                        "idx_cus_customer",
                        f"CREATE INDEX idx_cus_customer ON {target} ([{col_customer}])",
                    ),
                ]
            elif table_lower == "vnd":
                # Composite index for duplicate checking (vendor + vcode)
                indexes = [
                    (
                        "idx_vnd_vendor_vcode",
                        f"CREATE INDEX idx_vnd_vendor_vcode ON {target} ([{col_vendor}], [{col_vcode}])",
                    ),
                ]
            elif table_lower == "str":
                # Index for duplicate checking (store)
                indexes = [
                    (
                        "idx_str_store",
                        f"CREATE INDEX idx_str_store ON {target} ([{col_store}])",
                    ),
                ]
            elif table_lower == "cat":
                # Index for duplicate checking (cat)
                indexes = [
                    (
                        "idx_cat_cat",
                        f"CREATE INDEX idx_cat_cat ON {target} ([{col_cat}])",
                    ),
                ]
            elif table_lower == "emp":
                # Composite index for duplicate checking (id + uid)
                indexes = [
                    (
                        "idx_emp_id_uid",
                        f"CREATE INDEX idx_emp_id_uid ON {target} ([{col_id}], [{col_uid}])",
                    ),
                ]
            else:
                # No indexes for other tables
                indexes = []
        else:  # mysql
            target = f"`{safe_sql_name(table)}`"

            if table_lower == "jnh":
                # Indexes for jnh table (performance + duplicate checking)
                indexes = [
                    (
                        "idx_jnh_tstamp_sale",
                        f"CREATE INDEX idx_jnh_tstamp_sale ON {target} (`{col_tstamp}`, `{col_sale}`)",
                    ),
                    (
                        "idx_jnh_tstamp",
                        f"CREATE INDEX idx_jnh_tstamp ON {target} (`{col_tstamp}`)",
                    ),
                    (
                        "idx_jnh_sale",
                        f"CREATE INDEX idx_jnh_sale ON {target} (`{col_sale}`)",
                    ),
                ]
            elif table_lower == "jnl":
                # Indexes for jnl table (performance + duplicate checking)
                col_date = (
                    choose_column_name("date", existing_cols)
                    or choose_column_name("tdate", existing_cols)
                    or safe_sql_name("date")
                )
                indexes = [
                    (
                        "idx_jnl_sale_line_rflag",
                        f"CREATE INDEX idx_jnl_sale_line_rflag ON {target} (`{col_sale}`, `{col_line}`, `{col_rflag}`)",
                    ),
                    (
                        "idx_jnl_sale_rflag_sku",
                        f"CREATE INDEX idx_jnl_sale_rflag_sku ON {target} (`{col_sale}`, `{col_rflag}`, `{col_sku}`)",
                    ),
                    (
                        "idx_jnl_sale",
                        f"CREATE INDEX idx_jnl_sale ON {target} (`{col_sale}`)",
                    ),
                    (
                        "idx_jnl_date_rflag_sku",
                        f"CREATE INDEX idx_jnl_date_rflag_sku ON {target} (`{col_date}`, `{col_rflag}`, `{col_sku}`)",
                    ),
                    (
                        "idx_jnl_sku_rflag_date",
                        f"CREATE INDEX idx_jnl_sku_rflag_date ON {target} (`{col_sku}`, `{col_rflag}`, `{col_date}`)",
                    ),
                    (
                        "idx_jnl_line_rflag_sale",
                        f"CREATE INDEX idx_jnl_line_rflag_sale ON {target} (`{col_line}`, `{col_rflag}`, `{col_sale}`)",
                    ),
                ]
            elif table_lower in ("inv", "stk", "prc"):
                # Index for duplicate checking (SKU)
                indexes = [
                    (
                        f"idx_{table_lower}_sku",
                        f"CREATE INDEX idx_{table_lower}_sku ON {target} (`{col_sku}`)",
                    ),
                ]
            elif table_lower == "upc":
                # Composite index for duplicate checking (UPC + SKU)
                indexes = [
                    (
                        "idx_upc_upc_sku",
                        f"CREATE INDEX idx_upc_upc_sku ON {target} (`{col_upc}`, `{col_sku}`)",
                    ),
                ]
            elif table_lower == "hst":
                # Indexes for hst table (sales history)
                indexes = [
                    (
                        "idx_hst_sku_date",
                        f"CREATE INDEX idx_hst_sku_date ON {target} (`{col_sku}`, `{col_date}`)",
                    ),
                    (
                        "idx_hst_date_sku",
                        f"CREATE INDEX idx_hst_date_sku ON {target} (`{col_date}`, `{col_sku}`)",
                    ),
                    (
                        "idx_hst_date",
                        f"CREATE INDEX idx_hst_date ON {target} (`{col_date}`)",
                    ),
                ]
            elif table_lower in ("slh", "sll"):
                # Index for duplicate checking (listnum)
                indexes = [
                    (
                        f"idx_{table_lower}_listnum",
                        f"CREATE INDEX idx_{table_lower}_listnum ON {target} (`{col_listnum}`)",
                    ),
                ]
            elif table_lower == "poh":
                # Indexes for poh table: duplicate checking + performance for date/status/vendor queries
                # Find actual column names from existing columns
                col_rcvdate = (
                    choose_column_name("rcvdate", existing_cols)
                    or choose_column_name("received_date", existing_cols)
                    or choose_column_name("rcv_date", existing_cols)
                )
                col_status = choose_column_name(
                    "status", existing_cols
                ) or choose_column_name("stat", existing_cols)
                col_vendor = choose_column_name(
                    "vendor", existing_cols
                ) or choose_column_name("vcode", existing_cols)

                # Only create indexes if columns exist
                index_list = [
                    ("idx_poh_order", col_order),
                ]
                if col_rcvdate and col_status and col_vendor:
                    index_list.append(
                        (
                            "idx_poh_status_rcvdate_vendor",
                            f"{col_status}, {col_rcvdate}, {col_vendor}",
                        )
                    )
                if col_rcvdate:
                    index_list.append(("idx_poh_rcvdate", col_rcvdate))
                if col_vendor:
                    index_list.append(("idx_poh_vendor", col_vendor))

                indexes = [
                    (name, f"CREATE INDEX {name} ON {target} (`{cols}`)")
                    for name, cols in index_list
                ]
            elif table_lower == "pod":
                # Indexes for pod table (purchase order details)
                index_list = [
                    ("idx_pod_order", col_order),
                    ("idx_pod_sku", col_sku),
                ]
                if col_store:
                    index_list.append(("idx_pod_store", col_store))
                # Composite index for common query pattern: sku + order
                index_list.append(("idx_pod_sku_order", f"`{col_sku}`, `{col_order}`"))

                indexes = [
                    (name, f"CREATE INDEX {name} ON {target} ({cols})")
                    for name, cols in index_list
                ]
            elif table_lower == "glb":
                # Index for duplicate checking (date)
                indexes = [
                    (
                        "idx_glb_date",
                        f"CREATE INDEX idx_glb_date ON {target} (`{col_date}`)",
                    ),
                ]
            elif table_lower == "cnt":
                # Index for duplicate checking (code)
                indexes = [
                    (
                        "idx_cnt_code",
                        f"CREATE INDEX idx_cnt_code ON {target} (`{col_code}`)",
                    ),
                ]
            elif table_lower == "cus":
                # Index for duplicate checking (customer)
                indexes = [
                    (
                        "idx_cus_customer",
                        f"CREATE INDEX idx_cus_customer ON {target} (`{col_customer}`)",
                    ),
                ]
            elif table_lower == "vnd":
                # Composite index for duplicate checking (vendor + vcode)
                indexes = [
                    (
                        "idx_vnd_vendor_vcode",
                        f"CREATE INDEX idx_vnd_vendor_vcode ON {target} (`{col_vendor}`, `{col_vcode}`)",
                    ),
                ]
            elif table_lower == "str":
                # Index for duplicate checking (store)
                indexes = [
                    (
                        "idx_str_store",
                        f"CREATE INDEX idx_str_store ON {target} (`{col_store}`)",
                    ),
                ]
            elif table_lower == "cat":
                # Index for duplicate checking (cat)
                indexes = [
                    (
                        "idx_cat_cat",
                        f"CREATE INDEX idx_cat_cat ON {target} (`{col_cat}`)",
                    ),
                ]
            elif table_lower == "emp":
                # Composite index for duplicate checking (id + uid)
                indexes = [
                    (
                        "idx_emp_id_uid",
                        f"CREATE INDEX idx_emp_id_uid ON {target} (`{col_id}`, `{col_uid}`)",
                    ),
                ]
            else:
                # No indexes for other tables
                indexes = []

        # Create each index (ignore errors if index already exists)
        for idx_name, idx_sql in indexes:
            try:
                cur.execute(idx_sql)
                log_to_gui(f"Created index {idx_name} on {table}")
            except Exception as e:
                # Index might already exist, or column might not exist yet
                # Log but don't fail - this is expected in some cases
                error_msg = str(e).lower()
                if (
                    "already exists" in error_msg
                    or "duplicate" in error_msg
                    or "duplicate key" in error_msg
                ):
                    log_to_gui(f"Index {idx_name} already exists on {table}, skipping")
                else:
                    log_to_gui(
                        f"WARNING: Could not create index {idx_name} on {table}: {e}"
                    )

        conn.commit()
    except Exception as e:
        log_to_gui(f"WARNING: Error creating indexes for {table}: {e}")
        # Don't fail table creation if indexes fail
        try:
            conn.rollback()
        except Exception:
            pass


def ensure_table(
    conn, engine: str, table: str, fields, schema: str = None, recreate: bool = False
):
    if recreate and table_exists(conn, engine, table, schema):
        drop_table(conn, engine, table, schema)
    if not table_exists(conn, engine, table, schema):
        ddl = build_create_table_sql(table, fields, engine, schema)
        cur = conn.cursor()
        cur.execute(ddl)
        conn.commit()

        # Create indexes for jnh and jnl tables after table creation
        create_table_indexes(conn, engine, table, schema)
        return
    if not recreate:
        existing = existing_columns(conn, engine, table, schema)
        for f in fields:
            chosen = choose_column_name(f.name, existing)
            if chosen.lower() not in {c.lower() for c in existing}:
                add_missing_column(
                    conn, engine, table, safe_sql_name(f.name), f, schema
                )
                existing.append(safe_sql_name(f.name))


def truncate_table(conn, engine: str, table: str, schema: str = None):
    cur = conn.cursor()
    if engine == "mssql":
        target = (
            f"[{schema}].[{safe_sql_name(table)}]"
            if schema
            else f"[{safe_sql_name(table)}]"
        )
        cur.execute(f"TRUNCATE TABLE {target}")
    else:
        target = f"`{safe_sql_name(table)}`"
        cur.execute(f"TRUNCATE TABLE {target}")
    conn.commit()


def coerce_value(v):
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v.decode("latin-1", errors="replace")
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (datetime, date)):
        return v
    if isinstance(v, bool):
        return 1 if v else 0
    return v


def iter_dbf_rows(
    dbf_path: str,
    date_field: Optional[str] = None,
    since_date: Optional[datetime] = None,
    date_range_start: Optional[datetime] = None,
    date_range_end: Optional[datetime] = None,
    related_table_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Iterable[List[Any]]]:
    """
    Iterate DBF rows, optionally filtering by date field.

    Args:
        dbf_path: Path to DBF file
        date_field: Field name to use for date filtering (case-insensitive)
        since_date: Only include records with date_field >= since_date (delta sync)
        date_range_start: Only include records with date_field >= date_range_start (initial load filter)
        date_range_end: Only include records with date_field <= date_range_end (initial load filter)
        related_table_config: Optional dict with keys:
            - related_table: name of related table (without .dbf)
            - join_field_local: field in current table to match on
            - join_field_related: field in related table to match on
            - date_field_related: date field in related table to check
    """
    try:
        table = open_dbf(dbf_path)
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid date" in error_msg or (
            "date" in error_msg and ("b'\\x00" in str(e) or "\\x00" in str(e))
        ):
            log_to_gui(
                f"WARNING: Error opening DBF file (date field issue): {e}. Attempting to continue with defensive parsing..."
            )
            # Try to open with load=False (which we already do) - the issue is likely during iteration
            # Re-raise to let caller handle it, but log the warning
            raise RuntimeError(f"Cannot open DBF file due to date field error: {e}")
        raise

    field_names = [f.name for f in table.fields]

    # Handle related table filtering (e.g., jnl using jnh.tstamp, sll using slh.tstamp)
    valid_join_values = None
    join_field_local_idx = None
    if related_table_config and (since_date or date_range_start):
        # Load related table to get valid join values based on its date field
        related_table_name = related_table_config.get("related_table")
        join_field_local = related_table_config.get("join_field_local")
        join_field_related = related_table_config.get("join_field_related")
        date_field_related = related_table_config.get("date_field_related")

        if (
            related_table_name
            and join_field_local
            and join_field_related
            and date_field_related
        ):
            # Find join field index in current table
            for i, name in enumerate(field_names):
                if name.lower() == join_field_local.lower():
                    join_field_local_idx = i
                    break

            if join_field_local_idx is not None:
                # Load related table from same folder
                folder = os.path.dirname(dbf_path)
                related_dbf_path = os.path.join(folder, f"{related_table_name}.dbf")

                if os.path.exists(related_dbf_path):
                    try:
                        # Load related table and extract valid join values
                        related_table = open_dbf(related_dbf_path)
                        related_field_names = [f.name for f in related_table.fields]

                        # Find indices in related table
                        join_field_related_idx = None
                        date_field_related_idx = None

                        for i, name in enumerate(related_field_names):
                            if name.lower() == join_field_related.lower():
                                join_field_related_idx = i
                            if name.lower() == date_field_related.lower():
                                date_field_related_idx = i

                        if (
                            join_field_related_idx is not None
                            and date_field_related_idx is not None
                        ):
                            # Determine effective start date
                            effective_start = None
                            if since_date and date_range_start:
                                since = (
                                    since_date.replace(tzinfo=None)
                                    if since_date.tzinfo
                                    else since_date
                                )
                                range_start = (
                                    date_range_start.replace(tzinfo=None)
                                    if date_range_start.tzinfo
                                    else date_range_start
                                )
                                effective_start = max(since, range_start)
                            elif since_date:
                                effective_start = (
                                    since_date.replace(tzinfo=None)
                                    if since_date.tzinfo
                                    else since_date
                                )
                            elif date_range_start:
                                effective_start = (
                                    date_range_start.replace(tzinfo=None)
                                    if date_range_start.tzinfo
                                    else date_range_start
                                )

                            if effective_start:
                                # Extract valid join values from related table
                                # Use both string and numeric sets for efficient lookup
                                valid_join_values_str = set()
                                valid_join_values_num = set()

                                # Progress tracking for large tables
                                # Try to get record count (DBF objects may not support len())
                                try:
                                    total_records = len(related_table)
                                except (TypeError, AttributeError):
                                    # Try alternative method to get count
                                    try:
                                        total_records = (
                                            related_table.record_count
                                            if hasattr(related_table, "record_count")
                                            else None
                                        )
                                    except Exception:
                                        total_records = None

                                processed_count = 0
                                progress_interval = max(
                                    1000,
                                    (total_records // 20) if total_records else 1000,
                                )

                                if total_records:
                                    log_to_gui(
                                        f"Loading related table '{related_table_name}' ({total_records:,} records) to build filter set..."
                                    )
                                else:
                                    log_to_gui(
                                        f"Loading related table '{related_table_name}' to build filter set..."
                                    )

                                for rec in related_table:
                                    try:
                                        processed_count += 1
                                        # Progress logging for large tables
                                        if (
                                            processed_count % progress_interval == 0
                                            and total_records
                                        ):
                                            pct = processed_count / total_records * 100
                                            log_to_gui(
                                                f"  Processing {related_table_name}: {processed_count:,}/{total_records:,} ({pct:.1f}%)..."
                                            )
                                        elif processed_count % progress_interval == 0:
                                            log_to_gui(
                                                f"  Processing {related_table_name}: {processed_count:,} records..."
                                            )

                                        join_val = rec.get(
                                            related_field_names[join_field_related_idx]
                                        )
                                        date_val = rec.get(
                                            related_field_names[date_field_related_idx]
                                        )

                                        # Skip deleted records (DBF files mark deleted records)
                                        if hasattr(rec, "deleted") and rec.deleted:
                                            continue

                                        # Parse date value - early exit on None
                                        if date_val is None:
                                            continue

                                        # Handle bytes, strings, date objects
                                        if isinstance(date_val, bytes):
                                            if (
                                                all(b == 0 for b in date_val)
                                                or len(date_val) == 0
                                            ):
                                                continue
                                            try:
                                                for enc in (
                                                    "utf-8",
                                                    "latin-1",
                                                    "cp1252",
                                                    "cp437",
                                                    "ascii",
                                                ):
                                                    try:
                                                        date_val = date_val.decode(
                                                            enc
                                                        ).strip()
                                                        break
                                                    except (
                                                        UnicodeDecodeError,
                                                        AttributeError,
                                                    ):
                                                        continue
                                                if (
                                                    isinstance(date_val, bytes)
                                                    or not date_val
                                                ):
                                                    continue
                                            except Exception:
                                                continue

                                        if isinstance(date_val, str):
                                            date_val = date_val.strip()
                                            if not date_val:
                                                continue

                                        # Try parsing common date formats
                                        parsed_date = None
                                        if isinstance(date_val, (date, datetime)):
                                            parsed_date = date_val
                                            if isinstance(
                                                parsed_date, date
                                            ) and not isinstance(parsed_date, datetime):
                                                parsed_date = datetime.combine(
                                                    parsed_date, datetime.min.time()
                                                )
                                        else:
                                            # Try strptime with common formats
                                            # Format: "11/04/2025 07:58:47 PM" - MM/DD/YYYY HH:MM:SS AM/PM
                                            date_formats = [
                                                "%m/%d/%Y %I:%M:%S %p",  # MM/DD/YYYY HH:MM:SS AM/PM (e.g., 11/04/2025 07:58:47 PM)
                                                "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS (24-hour)
                                                "%Y-%m-%d",
                                                "%Y-%m-%d %H:%M:%S",
                                                "%Y-%m-%d %H:%M:%S.%f",
                                                "%m/%d/%Y",  # MM/DD/YYYY (date only)
                                                "%m-%d-%Y",
                                                "%d/%m/%Y",
                                                "%d-%m-%Y",
                                                "%Y%m%d",
                                                "%Y%m%d%H%M%S",
                                            ]
                                            for fmt in date_formats:
                                                try:
                                                    parsed_date = datetime.strptime(
                                                        str(date_val), fmt
                                                    )
                                                    break
                                                except (ValueError, TypeError):
                                                    continue

                                            # Try dateutil as fallback
                                            if parsed_date is None:
                                                try:
                                                    from dateutil import (
                                                        parser as date_parser,
                                                    )

                                                    parsed_date = date_parser.parse(
                                                        str(date_val), fuzzy=False
                                                    )
                                                except (
                                                    ImportError,
                                                    ValueError,
                                                    TypeError,
                                                ):
                                                    pass

                                        if parsed_date and isinstance(
                                            parsed_date, datetime
                                        ):
                                            # Normalize timezone
                                            if parsed_date.tzinfo is not None:
                                                parsed_date = parsed_date.replace(
                                                    tzinfo=None
                                                )

                                            # Check if date is newer than effective_start
                                            if parsed_date >= effective_start:
                                                # Add join value to both string and numeric sets for efficient lookup
                                                if join_val is not None:
                                                    # Try to keep as numeric if possible (faster comparisons)
                                                    try:
                                                        if isinstance(
                                                            join_val,
                                                            (int, float, Decimal),
                                                        ):
                                                            valid_join_values_num.add(
                                                                join_val
                                                            )
                                                        # Also add as string for mixed-type matching
                                                        valid_join_values_str.add(
                                                            str(join_val).strip()
                                                        )
                                                    except Exception:
                                                        # Fallback to string only
                                                        valid_join_values_str.add(
                                                            str(join_val).strip()
                                                            if join_val
                                                            else ""
                                                        )
                                    except Exception:
                                        continue  # Skip problematic rows in related table

                                # Combine both sets for efficient lookup
                                valid_join_values = {
                                    "str": valid_join_values_str,
                                    "num": valid_join_values_num,
                                }

                                # Log the related table filtering result
                                # Note: numeric values are in both sets, so we count unique values
                                total_valid = len(
                                    valid_join_values_str
                                )  # String set contains all values
                                log_msg = f"Related table '{related_table_name}': Found {total_valid} valid {join_field_related} values with {date_field_related} >= {effective_start}"
                                log_to_gui(log_msg)

                                # If no valid values found, this is expected when there are no new sales
                                if total_valid == 0:
                                    log_to_gui(
                                        f"  INFO: No new records found in '{related_table_name}' with {date_field_related} >= {effective_start}"
                                    )
                                    log_to_gui(
                                        f"  This is expected when there are no new sales since the last sync. Will insert 0 rows for related table."
                                    )
                    except Exception as e:
                        error_msg = f"WARNING: Could not load related table '{related_table_name}': {e}. Falling back to full sync."
                        log_to_gui(error_msg)
                        valid_join_values = None

    # Find date field index (case-insensitive)
    date_field_idx = None
    needs_date_filter = bool(
        date_field and (since_date or date_range_start or date_range_end)
    )

    if needs_date_filter:
        for i, name in enumerate(field_names):
            if name.lower() == date_field.lower():
                date_field_idx = i
                break

        if date_field_idx is None:
            import warnings
            import sys

            warning_msg = f"WARNING: date_field '{date_field}' not found in DBF '{os.path.basename(dbf_path)}'. Available fields: {', '.join(field_names[:10])}{'...' if len(field_names) > 10 else ''}. Date filtering disabled."
            warnings.warn(warning_msg)
            # Also print to stderr for visibility in headless mode
            print(warning_msg, file=sys.stderr)

    def validate_parsed_date(parsed_dt: datetime) -> bool:
        """Validate that a parsed date has a reasonable year range."""
        if not isinstance(parsed_dt, datetime):
            return False
        year = parsed_dt.year
        # Reject dates with years outside reasonable range (1900-2100)
        if year < 1900 or year > 2100:
            return False
        return True

    def gen():
        # Wrap the table iteration itself to catch errors from DBF library
        # The dbfread library may raise exceptions when encountering invalid date fields
        # We need to catch these at the iteration level
        skipped_count = 0
        max_skipped_warnings = 10  # Only log first 10 warnings to avoid spam

        # Create an iterator from the DBF table
        try:
            table_iter = iter(table)
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid date" in error_msg or (
                "date" in error_msg and ("b'\\x00" in str(e) or "\\x00" in str(e))
            ):
                log_to_gui(
                    f"WARNING: Error creating table iterator (date field issue): {e}. Will attempt to skip problematic records."
                )
                # Try to get a safe iterator - if this fails, we can't proceed
                try:
                    table_iter = iter(table)
                except Exception:
                    log_to_gui(
                        f"ERROR: Cannot iterate DBF file due to date field errors. Skipping this file."
                    )
                    return  # Return empty generator
            else:
                raise

        while True:
            try:
                # Try to get next record - this may raise an exception if date fields are invalid
                rec = next(table_iter)
            except StopIteration:
                # End of table reached
                break
            except Exception as e:
                # Catch any exception from DBF library during record access
                error_msg = str(e).lower()
                error_str = str(e)
                if "invalid date" in error_msg or (
                    "date" in error_msg
                    and (
                        "b'\\x00" in error_str
                        or "\\x00" in error_str
                        or "null" in error_msg
                    )
                ):
                    skipped_count += 1
                    if skipped_count <= max_skipped_warnings:
                        log_to_gui(
                            f"  Skipping record with invalid date field: {error_str[:100]}"
                        )
                    elif skipped_count == max_skipped_warnings + 1:
                        log_to_gui(f"  ... (suppressing further date field warnings)")
                    # Skip this record and continue
                    continue
                else:
                    # For non-date errors, also skip but log differently
                    skipped_count += 1
                    if skipped_count <= max_skipped_warnings:
                        log_to_gui(f"  Skipping record due to error: {error_str[:100]}")
                    continue
            try:
                # Try to get the raw record first, handling date field errors specially
                # Wrap the entire record access in a try-except to catch DBF library errors
                try:
                    # Attempt to access all fields - this may raise an error if date fields are invalid
                    row = []
                    for name in field_names:
                        try:
                            val = rec.get(name)
                            # Check if this is a date field and handle it specially
                            if (
                                date_field_idx is not None
                                and name.lower() == date_field.lower()
                            ):
                                # Special handling for date fields to catch DBF library errors
                                try:
                                    val = coerce_value(val)
                                    # Check for null bytes in date field before processing
                                    if isinstance(val, bytes):
                                        # Check if all bytes are null (uninitialized date)
                                        if all(b == 0 for b in val) or len(val) == 0:
                                            val = None  # Mark as None to skip later
                                    # Additional validation: check if it's a date/datetime with invalid year
                                    elif isinstance(val, (date, datetime)):
                                        if not validate_parsed_date(
                                            val
                                            if isinstance(val, datetime)
                                            else datetime.combine(
                                                val, datetime.min.time()
                                            )
                                        ):
                                            val = None  # Mark invalid dates as None
                                except Exception:
                                    # DBF library couldn't parse this date field - mark as None
                                    val = None
                            else:
                                val = coerce_value(val)
                            row.append(val)
                        except Exception as e:
                            # If this is the date field and we get an error, mark it as None
                            error_msg = str(e).lower()
                            if (
                                date_field_idx is not None
                                and name.lower() == date_field.lower()
                            ):
                                if "invalid date" in error_msg or "date" in error_msg:
                                    row.append(None)  # Mark date field as None to skip
                                    continue
                            # For other fields, try to use a default value or skip the field
                            row.append(None)
                except Exception as e:
                    # Handle errors from DBF library when accessing record (e.g., invalid date fields)
                    # This catches errors that occur when the library tries to parse date fields
                    error_msg = str(e).lower()
                    error_str = str(e)
                    if "invalid date" in error_msg or (
                        "date" in error_msg
                        and (
                            "b'\\x00" in error_str
                            or "null" in error_msg
                            or "\\x00" in error_str
                        )
                    ):
                        # Silently skip rows with invalid dates - this is expected for some DBF files
                        continue
                    # Re-raise if it's not a date error - we want to handle it at outer level
                    raise
            except Exception as e:
                # Handle errors from DBF library when reading entire record (e.g., invalid date fields)
                # Skip this row and continue with next record
                error_msg = str(e).lower()
                error_str = str(e)
                if "invalid date" in error_msg or (
                    "date" in error_msg
                    and (
                        "b'\\x00" in error_str
                        or "null" in error_msg
                        or "\\x00" in error_str
                    )
                ):
                    # Silently skip rows with invalid dates - this is expected for some DBF files
                    continue
                # For other errors, also skip but this is less common
                continue

            # Apply date filter if specified
            if date_field_idx is not None and needs_date_filter:
                try:
                    date_val = row[date_field_idx]
                except (IndexError, KeyError, TypeError):
                    # Skip row if we can't access the date field
                    continue
                except Exception:
                    # Handle any other errors accessing date field (e.g., invalid date from DBF library)
                    # Skip this row rather than failing the entire sync
                    continue

                # Handle None, date, datetime, string dates, or invalid bytes
                if date_val is None:
                    continue

                # Handle bytes (common in DBF for uninitialized date fields)
                if isinstance(date_val, bytes):
                    # Check if all bytes are null (uninitialized date)
                    if all(b == 0 for b in date_val) or len(date_val) == 0:
                        continue  # Skip invalid null date
                    # Try to decode bytes to string first
                    try:
                        # Try common encodings
                        for enc in ("utf-8", "latin-1", "cp1252", "cp437", "ascii"):
                            try:
                                decoded = date_val.decode(enc).strip()
                                if decoded and not all(
                                    c in ("\x00", " ", "\t", "\n", "\r")
                                    for c in decoded
                                ):
                                    date_val = decoded
                                    break
                            except (UnicodeDecodeError, AttributeError):
                                continue
                        # If still bytes or couldn't decode, skip this row
                        if isinstance(date_val, bytes) or not date_val:
                            continue
                    except Exception:
                        continue  # Skip row if we can't decode

                # Handle string dates - try comprehensive list of formats
                if isinstance(date_val, str):
                    # Skip empty strings or whitespace-only strings
                    date_val = date_val.strip()
                    if not date_val or all(
                        c in (" ", "\t", "\n", "\r", "\x00") for c in date_val
                    ):
                        continue

                    # Validate date string before parsing (check for obviously invalid years)
                    # Check if the string contains a year that looks invalid (e.g., 4102, 9999, etc.)
                    if len(date_val) >= 4:
                        # Try to extract year from common positions
                        year_candidates = []
                        # Check for 4-digit years at start or end
                        if date_val[:4].isdigit():
                            year_candidates.append(int(date_val[:4]))
                        if date_val[-4:].isdigit():
                            year_candidates.append(int(date_val[-4:]))
                        # Check for years in middle (e.g., MM/DD/YYYY)
                        parts = date_val.replace("/", "-").replace(".", "-").split()
                        if parts:
                            for part in parts[0].split("-"):
                                if len(part) == 4 and part.isdigit():
                                    year_candidates.append(int(part))

                        # Reject if any year is outside reasonable range (1900-2100)
                        if year_candidates:
                            if any(
                                year < 1900 or year > 2100 for year in year_candidates
                            ):
                                continue  # Skip row with invalid year

                    # Try parsing comprehensive list of date formats
                    date_formats = [
                        # ISO formats
                        "%Y-%m-%d",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S.%f",
                        # US formats
                        "%m/%d/%Y",
                        "%m/%d/%Y %H:%M:%S",
                        "%m-%d-%Y",
                        "%m-%d-%Y %H:%M:%S",
                        "%m.%d.%Y",
                        # European formats
                        "%d/%m/%Y",
                        "%d/%m/%Y %H:%M:%S",
                        "%d-%m-%Y",
                        "%d-%m-%Y %H:%M:%S",
                        "%d.%m.%Y",
                        # Compact formats
                        "%Y%m%d",
                        "%d%m%Y",
                        "%m%d%Y",
                        "%Y%m%d%H%M%S",
                        # Slash formats
                        "%Y/%m/%d",
                        "%Y/%m/%d %H:%M:%S",
                        "%d/%m/%Y",
                        "%m/%d/%Y",
                        # DateParse formats (flexible)
                        "%d %b %Y",
                        "%d %B %Y",
                        "%b %d, %Y",
                        "%B %d, %Y",
                        "%Y-%m-%d %H:%M",
                        "%d-%m-%Y %H:%M",
                        "%m-%d-%Y %H:%M",
                    ]

                    parsed = False
                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_val, fmt)
                            # Validate the parsed date has a reasonable year
                            if validate_parsed_date(parsed_date):
                                date_val = parsed_date
                                parsed = True
                                break
                        except (ValueError, TypeError):
                            continue

                    # If still not parsed, try dateutil parser as fallback (if available)
                    if not parsed:
                        try:
                            from dateutil import parser as date_parser

                            parsed_date = date_parser.parse(date_val, fuzzy=False)
                            if validate_parsed_date(parsed_date):
                                date_val = parsed_date
                                parsed = True
                        except (ImportError, ValueError, TypeError):
                            pass

                    if not parsed or isinstance(date_val, str):
                        continue  # Couldn't parse or validate, skip this row

                # Handle date objects (convert to datetime)
                if isinstance(date_val, date) and not isinstance(date_val, datetime):
                    try:
                        date_val = datetime.combine(date_val, datetime.min.time())
                        # Validate the converted date has a reasonable year
                        if not validate_parsed_date(date_val):
                            continue  # Skip row with invalid year
                    except Exception:
                        continue  # Skip row if conversion fails

                # Final check: must be datetime at this point
                if not isinstance(date_val, datetime):
                    continue  # Skip row if still not a datetime

                # Final validation: ensure year is reasonable
                if not validate_parsed_date(date_val):
                    continue  # Skip row with invalid year

                # Normalize timezone for comparison (convert timezone-aware to naive)
                if date_val.tzinfo is not None:
                    date_val = date_val.replace(tzinfo=None)

                # Apply date range filters
                # Determine effective start date (use max of since_date and date_range_start)
                effective_start = None
                if since_date and date_range_start:
                    # Normalize timezone for since_date
                    since = (
                        since_date.replace(tzinfo=None)
                        if since_date.tzinfo
                        else since_date
                    )
                    range_start = (
                        date_range_start.replace(tzinfo=None)
                        if date_range_start.tzinfo
                        else date_range_start
                    )
                    effective_start = max(since, range_start)
                elif since_date:
                    effective_start = (
                        since_date.replace(tzinfo=None)
                        if since_date.tzinfo
                        else since_date
                    )
                elif date_range_start:
                    effective_start = (
                        date_range_start.replace(tzinfo=None)
                        if date_range_start.tzinfo
                        else date_range_start
                    )

                # Check start date
                if effective_start and date_val < effective_start:
                    continue

                range_end = None
                if date_range_end:
                    range_end = (
                        date_range_end.replace(tzinfo=None)
                        if date_range_end.tzinfo
                        else date_range_end
                    )
                if range_end and date_val > range_end:
                    continue

            # Apply related table filtering if configured
            if valid_join_values is not None and join_field_local_idx is not None:
                try:
                    join_val_raw = row[join_field_local_idx]
                    if join_val_raw is None:
                        continue  # Skip rows with null join values

                    # Try numeric comparison first (faster), then string
                    join_val_str = None
                    join_val_num = None

                    if isinstance(join_val_raw, (int, float, Decimal)):
                        join_val_num = join_val_raw
                    else:
                        join_val_str = str(join_val_raw).strip()

                    # Check both numeric and string sets
                    if isinstance(valid_join_values, dict):
                        # New optimized format with separate numeric/string sets
                        found = False
                        if (
                            join_val_num is not None
                            and join_val_num in valid_join_values.get("num", set())
                        ):
                            found = True
                        elif join_val_str and join_val_str in valid_join_values.get(
                            "str", set()
                        ):
                            found = True
                        if not found:
                            continue  # Skip row if join value not in valid set
                    else:
                        # Legacy format (backward compatibility)
                        join_val = (
                            join_val_str or str(join_val_num)
                            if join_val_num is not None
                            else None
                        )
                        if join_val and join_val not in valid_join_values:
                            continue  # Skip row if join value not in valid set
                except (IndexError, TypeError):
                    continue  # Skip row if can't access join field

            yield row

        # Log summary if we skipped records during iteration
        if skipped_count > 0:
            log_to_gui(
                f"  Skipped {skipped_count} record(s) with invalid date fields or other errors"
            )

    return field_names, gen()


def build_insert_sql(
    table: str,
    col_names: List[str],
    engine: str,
    schema: str = None,
    existing: List[str] = None,
) -> Tuple[str, List[str]]:
    if existing is None:
        dest_cols = [safe_sql_name(c) for c in col_names]
    else:
        dest_cols = [choose_column_name(c, existing) for c in col_names]

    if engine == "mssql":
        target = (
            f"[{schema}].[{safe_sql_name(table)}]"
            if schema
            else f"[{safe_sql_name(table)}]"
        )
        cols = ", ".join(f"[{c}]" for c in dest_cols)
        placeholders = ", ".join(["?"] * len(dest_cols))
        return f"INSERT INTO {target} ({cols}) VALUES ({placeholders})", dest_cols
    else:
        target = f"`{safe_sql_name(table)}`"
        cols = ", ".join(f"`{c}`" for c in dest_cols)
        placeholders = ", ".join(["%s"] * len(dest_cols))
        return f"INSERT INTO {target} ({cols}) VALUES ({placeholders})", dest_cols


def get_existing_keys(
    conn,
    engine: str,
    table: str,
    table_lower: str,
    col_names: List[str],
    existing_cols: List[str],
    schema: str = None,
    recreate: bool = False,
) -> set:
    """
    Get existing keys from table to prevent duplicates.
    Returns a set of tuples representing unique keys.

    Args:
        col_names: DBF column names (from iter_dbf_rows)
        existing_cols: Actual database column names (from existing_columns)
    """
    if recreate or not table_exists(conn, engine, table, schema):
        return set()  # No existing keys if recreating or table doesn't exist

    cur = conn.cursor()
    existing_keys = set()

    try:
        # Map DBF column names to database column names
        def find_db_col(dbf_col_name: str) -> Optional[str]:
            """Find the actual database column name for a DBF column name."""
            dbf_lower = dbf_col_name.lower()
            # Try exact match first
            for db_col in existing_cols:
                if (
                    db_col.lower() == dbf_lower
                    or db_col.lower() == safe_sql_name(dbf_col_name).lower()
                ):
                    return db_col
            # Try to find by matching normalized names
            dbf_safe = safe_sql_name(dbf_col_name)
            for db_col in existing_cols:
                if db_col.lower() == dbf_safe.lower():
                    return db_col
            return None

        if table_lower in ("inv", "stk", "prc"):
            # Check for existing SKU
            sku_col = None
            for col in col_names:
                if col.lower() in ("sku", "item", "item_id"):
                    sku_col = find_db_col(col)
                    if sku_col:
                        break

            if sku_col:
                sku_col_safe = safe_sql_name(sku_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{sku_col_safe}] FROM {target} WHERE [{sku_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{sku_col_safe}` FROM {target} WHERE `{sku_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "upc":
            # Check for existing UPC + SKU combination
            upc_col = None
            sku_col = None
            for col in col_names:
                if col.lower() in ("upc", "barcode", "ean"):
                    upc_col = find_db_col(col)
                elif col.lower() in ("sku", "item", "item_id"):
                    sku_col = find_db_col(col)

            if upc_col and sku_col:
                upc_col_safe = safe_sql_name(upc_col)
                sku_col_safe = safe_sql_name(sku_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{upc_col_safe}], [{sku_col_safe}] FROM {target} WHERE [{upc_col_safe}] IS NOT NULL AND [{sku_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{upc_col_safe}`, `{sku_col_safe}` FROM {target} WHERE `{upc_col_safe}` IS NOT NULL AND `{sku_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None and row[1] is not None:
                        existing_keys.add((str(row[0]).strip(), str(row[1]).strip()))

        elif table_lower in ("jnh", "jnl"):
            # Check for existing sale
            sale_col = None
            for col in col_names:
                if col.lower() in ("sale", "sale_id", "invno", "invoice"):
                    sale_col = find_db_col(col)
                    if sale_col:
                        break

            if sale_col:
                sale_col_safe = safe_sql_name(sale_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{sale_col_safe}] FROM {target} WHERE [{sale_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{sale_col_safe}` FROM {target} WHERE `{sale_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower in ("poh", "pod"):
            # Check for existing order
            order_col = None
            for col in col_names:
                if col.lower() in (
                    "order",
                    "orderno",
                    "pono",
                    "po",
                    "order_num",
                    "order_number",
                ):
                    order_col = find_db_col(col)
                    if order_col:
                        break

            if order_col:
                order_col_safe = safe_sql_name(order_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{order_col_safe}] FROM {target} WHERE [{order_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{order_col_safe}` FROM {target} WHERE `{order_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "glb":
            # Check for existing date
            date_col = None
            for col in col_names:
                if col.lower() in (
                    "date",
                    "tdate",
                    "sale_date",
                    "trans_date",
                    "transaction_date",
                    "glb_date",
                ):
                    date_col = find_db_col(col)
                    if date_col:
                        break

            if date_col:
                date_col_safe = safe_sql_name(date_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    # For date comparison, use CAST to DATE to normalize to date only
                    sql = f"SELECT DISTINCT CAST([{date_col_safe}] AS DATE) FROM {target} WHERE [{date_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    # For date comparison, use DATE() function to normalize to date only
                    sql = f"SELECT DISTINCT DATE(`{date_col_safe}`) FROM {target} WHERE `{date_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        # Convert date to string for comparison (YYYY-MM-DD format)
                        if isinstance(row[0], (date, datetime)):
                            date_str = row[0].strftime("%Y-%m-%d")
                        else:
                            date_str = str(row[0])[
                                :10
                            ]  # Take first 10 chars (YYYY-MM-DD)
                        existing_keys.add((date_str,))

        elif table_lower == "hst":
            # Check for existing SKU + date combination
            sku_col = None
            date_col = None
            for col in col_names:
                if col.lower() in ("sku", "item", "item_id"):
                    sku_col = find_db_col(col)
                elif col.lower() in (
                    "date",
                    "tdate",
                    "sale_date",
                    "trans_date",
                    "transaction_date",
                ):
                    date_col = find_db_col(col)

            if sku_col and date_col:
                sku_col_safe = safe_sql_name(sku_col)
                date_col_safe = safe_sql_name(date_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    # For date comparison, use CAST to DATE to normalize to date only
                    sql = f"SELECT [{sku_col_safe}], CAST([{date_col_safe}] AS DATE) FROM {target} WHERE [{sku_col_safe}] IS NOT NULL AND [{date_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    # For date comparison, use DATE() function to normalize to date only
                    sql = f"SELECT `{sku_col_safe}`, DATE(`{date_col_safe}`) FROM {target} WHERE `{sku_col_safe}` IS NOT NULL AND `{date_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None and row[1] is not None:
                        # Convert date to string for comparison (YYYY-MM-DD format)
                        if isinstance(row[1], (date, datetime)):
                            date_str = row[1].strftime("%Y-%m-%d")
                        else:
                            date_str = str(row[1])[
                                :10
                            ]  # Take first 10 chars (YYYY-MM-DD)
                        existing_keys.add((str(row[0]).strip(), date_str))

        elif table_lower in ("slh", "sll"):
            # Check for existing listnum
            listnum_col = None
            for col in col_names:
                if col.lower() in ("listnum", "list_num", "list_number", "list"):
                    listnum_col = find_db_col(col)
                    if listnum_col:
                        break

            if listnum_col:
                listnum_col_safe = safe_sql_name(listnum_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{listnum_col_safe}] FROM {target} WHERE [{listnum_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{listnum_col_safe}` FROM {target} WHERE `{listnum_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "cnt":
            # Check for existing code
            code_col = None
            for col in col_names:
                if col.lower() in ("code", "code_id", "cnt_code"):
                    code_col = find_db_col(col)
                    if code_col:
                        break

            if code_col:
                code_col_safe = safe_sql_name(code_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{code_col_safe}] FROM {target} WHERE [{code_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{code_col_safe}` FROM {target} WHERE `{code_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "cus":
            # Check for existing customer
            customer_col = None
            for col in col_names:
                if col.lower() in ("customer", "cust", "customer_id", "cus_id"):
                    customer_col = find_db_col(col)
                    if customer_col:
                        break

            if customer_col:
                customer_col_safe = safe_sql_name(customer_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{customer_col_safe}] FROM {target} WHERE [{customer_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{customer_col_safe}` FROM {target} WHERE `{customer_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "vnd":
            # Check for existing vendor + vcode combination
            vendor_col = None
            vcode_col = None
            for col in col_names:
                if col.lower() in ("vendor", "vnd", "vendor_id", "vnd_id"):
                    vendor_col = find_db_col(col)
                elif col.lower() in ("vcode", "vendor_code", "vnd_code"):
                    vcode_col = find_db_col(col)

            if vendor_col and vcode_col:
                vendor_col_safe = safe_sql_name(vendor_col)
                vcode_col_safe = safe_sql_name(vcode_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{vendor_col_safe}], [{vcode_col_safe}] FROM {target} WHERE [{vendor_col_safe}] IS NOT NULL AND [{vcode_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{vendor_col_safe}`, `{vcode_col_safe}` FROM {target} WHERE `{vendor_col_safe}` IS NOT NULL AND `{vcode_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None and row[1] is not None:
                        existing_keys.add((str(row[0]).strip(), str(row[1]).strip()))

        elif table_lower == "str":
            # Check for existing store
            store_col = None
            for col in col_names:
                if col.lower() in (
                    "store",
                    "store_id",
                    "store_num",
                    "store_number",
                    "str_store",
                ):
                    store_col = find_db_col(col)
                    if store_col:
                        break

            if store_col:
                store_col_safe = safe_sql_name(store_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{store_col_safe}] FROM {target} WHERE [{store_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{store_col_safe}` FROM {target} WHERE `{store_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "cat":
            # Check for existing cat
            cat_col = None
            for col in col_names:
                if col.lower() in ("cat", "category", "cat_id", "category_id"):
                    cat_col = find_db_col(col)
                    if cat_col:
                        break

            if cat_col:
                cat_col_safe = safe_sql_name(cat_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{cat_col_safe}] FROM {target} WHERE [{cat_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{cat_col_safe}` FROM {target} WHERE `{cat_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None:
                        existing_keys.add((str(row[0]).strip(),))

        elif table_lower == "emp":
            # Check for existing id + uid combination
            id_col = None
            uid_col = None
            for col in col_names:
                if col.lower() in ("id", "emp_id", "employee_id", "eid"):
                    id_col = find_db_col(col)
                elif col.lower() in ("uid", "user_id", "userid", "emp_uid"):
                    uid_col = find_db_col(col)

            if id_col and uid_col:
                id_col_safe = safe_sql_name(id_col)
                uid_col_safe = safe_sql_name(uid_col)
                if engine == "mssql":
                    target = (
                        f"[{schema}].[{safe_sql_name(table)}]"
                        if schema
                        else f"[{safe_sql_name(table)}]"
                    )
                    sql = f"SELECT [{id_col_safe}], [{uid_col_safe}] FROM {target} WHERE [{id_col_safe}] IS NOT NULL AND [{uid_col_safe}] IS NOT NULL"
                else:
                    target = f"`{safe_sql_name(table)}`"
                    sql = f"SELECT `{id_col_safe}`, `{uid_col_safe}` FROM {target} WHERE `{id_col_safe}` IS NOT NULL AND `{uid_col_safe}` IS NOT NULL"

                cur.execute(sql)
                for row in cur.fetchall():
                    if row[0] is not None and row[1] is not None:
                        existing_keys.add((str(row[0]).strip(), str(row[1]).strip()))

    except Exception as e:
        log_to_gui(
            f"WARNING: Could not check existing keys for {table}: {e}. Proceeding without duplicate check."
        )
        return set()

    return existing_keys


def bulk_insert(
    conn,
    engine: str,
    table: str,
    dbf_path: str,
    batch_size: int = 1000,
    schema: str = None,
    recreate: bool = False,
    date_field: Optional[str] = None,
    since_date: Optional[datetime] = None,
    date_range_start: Optional[datetime] = None,
    date_range_end: Optional[datetime] = None,
    related_table_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """
    Bulk insert DBF data into RDS table.

    Args:
        date_field: Field name to use for date filtering
        since_date: Only insert records newer than this date (delta sync)
        date_range_start: Only insert records >= this date (initial load filter)
        date_range_end: Only insert records <= this date (initial load filter)
    """
    try:
        dbf_obj = open_dbf(dbf_path)
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid date" in error_msg or "date" in error_msg:
            log_to_gui(
                f"WARNING: Could not open DBF file (date field error): {e}. Attempting to continue with defensive parsing..."
            )
            # Try to open with a different encoding or skip problematic date fields
            # For now, we'll try to continue - the iter_dbf_rows function should handle it
            try:
                dbf_obj = open_dbf(dbf_path)
            except Exception:
                # If we can't open at all, re-raise - this is a critical error
                raise RuntimeError(f"Cannot open DBF file {dbf_path}: {e}")
        else:
            raise

    try:
        ensure_table(conn, engine, table, dbf_obj.fields, schema, recreate=recreate)
    except Exception as e:
        log_to_gui(f"WARNING: Error ensuring table structure: {e}")
        raise

    cols_existing = existing_columns(conn, engine, table, schema)

    try:
        col_names, row_iter = iter_dbf_rows(
            dbf_path,
            date_field=date_field,
            since_date=since_date,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            related_table_config=related_table_config,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid date" in error_msg or (
            "date" in error_msg and ("b'\\x00" in str(e) or "\\x00" in str(e))
        ):
            log_to_gui(
                f"WARNING: Date field error when iterating DBF rows: {e}. Rows with invalid dates will be skipped."
            )
            # Try to continue with a fresh iteration - our error handling should catch bad rows
            # Re-raise if it's not a date-related error
            raise RuntimeError(f"Critical error when reading DBF file: {e}")
        raise

    # Get existing keys to prevent duplicates (only for specific tables and when not recreating)
    table_lower = table.lower()
    existing_keys = set()
    duplicate_check_enabled = False

    if not recreate and table_lower in (
        "inv",
        "stk",
        "prc",
        "upc",
        "jnh",
        "jnl",
        "poh",
        "pod",
        "glb",
        "hst",
        "slh",
        "sll",
        "cnt",
        "cus",
        "vnd",
        "str",
        "cat",
        "emp",
    ):
        duplicate_check_enabled = True
        log_to_gui(f"Checking for existing records in {table} to prevent duplicates...")
        existing_keys = get_existing_keys(
            conn, engine, table, table_lower, col_names, cols_existing, schema, recreate
        )
        if existing_keys:
            log_to_gui(
                f"Found {len(existing_keys)} existing record(s) in {table}. Duplicates will be skipped."
            )
        else:
            log_to_gui(
                f"No existing records found in {table}. All records will be inserted."
            )

    insert_sql, _dest_cols = build_insert_sql(
        table, col_names, engine, schema, cols_existing
    )

    cur = conn.cursor()
    buf = []
    inserted = 0
    batches = 0
    skipped_duplicates = 0

    try:
        for row in row_iter:
            # Check for duplicates if enabled
            if duplicate_check_enabled:
                is_duplicate = False
                try:
                    if table_lower in ("inv", "stk", "prc"):
                        # Check SKU
                        sku_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("sku", "item", "item_id"):
                                sku_idx = i
                                break
                        if sku_idx is not None and sku_idx < len(row):
                            sku_val = row[sku_idx]
                            if sku_val is not None:
                                key = (str(sku_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "upc":
                        # Check UPC + SKU combination
                        upc_idx = None
                        sku_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("upc", "barcode", "ean"):
                                upc_idx = i
                            elif col.lower() in ("sku", "item", "item_id"):
                                sku_idx = i
                        if (
                            upc_idx is not None
                            and sku_idx is not None
                            and upc_idx < len(row)
                            and sku_idx < len(row)
                        ):
                            upc_val = row[upc_idx]
                            sku_val = row[sku_idx]
                            if upc_val is not None and sku_val is not None:
                                key = (str(upc_val).strip(), str(sku_val).strip())
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower in ("jnh", "jnl"):
                        # Check sale
                        sale_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("sale", "sale_id", "invno", "invoice"):
                                sale_idx = i
                                break
                        if sale_idx is not None and sale_idx < len(row):
                            sale_val = row[sale_idx]
                            if sale_val is not None:
                                key = (str(sale_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower in ("poh", "pod"):
                        # Check order
                        order_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "order",
                                "orderno",
                                "pono",
                                "po",
                                "order_num",
                                "order_number",
                            ):
                                order_idx = i
                                break
                        if order_idx is not None and order_idx < len(row):
                            order_val = row[order_idx]
                            if order_val is not None:
                                key = (str(order_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "glb":
                        # Check date
                        date_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "date",
                                "tdate",
                                "sale_date",
                                "trans_date",
                                "transaction_date",
                                "glb_date",
                            ):
                                date_idx = i
                                break
                        if date_idx is not None and date_idx < len(row):
                            date_val = row[date_idx]
                            if date_val is not None:
                                # Normalize date to YYYY-MM-DD format for comparison
                                date_str = None
                                if isinstance(date_val, (date, datetime)):
                                    date_str = date_val.strftime("%Y-%m-%d")
                                elif isinstance(date_val, str):
                                    # Try to parse and normalize date string
                                    try:
                                        # Try common date formats
                                        for fmt in [
                                            "%Y-%m-%d",
                                            "%m/%d/%Y",
                                            "%Y/%m/%d",
                                            "%d/%m/%Y",
                                        ]:
                                            try:
                                                parsed = datetime.strptime(
                                                    date_val[:10], fmt
                                                )
                                                date_str = parsed.strftime("%Y-%m-%d")
                                                break
                                            except ValueError:
                                                continue
                                        if not date_str:
                                            # Fallback: take first 10 chars if it looks like a date
                                            if (
                                                len(date_val) >= 10
                                                and date_val[4] in ("-", "/")
                                                and date_val[7] in ("-", "/")
                                            ):
                                                date_str = date_val[:10].replace(
                                                    "/", "-"
                                                )
                                    except Exception:
                                        pass

                                if date_str:
                                    key = (date_str,)
                                    if key in existing_keys:
                                        is_duplicate = True

                    elif table_lower == "hst":
                        # Check SKU + date combination
                        sku_idx = None
                        date_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("sku", "item", "item_id"):
                                sku_idx = i
                            elif col.lower() in (
                                "date",
                                "tdate",
                                "sale_date",
                                "trans_date",
                                "transaction_date",
                            ):
                                date_idx = i

                        if (
                            sku_idx is not None
                            and date_idx is not None
                            and sku_idx < len(row)
                            and date_idx < len(row)
                        ):
                            sku_val = row[sku_idx]
                            date_val = row[date_idx]
                            if sku_val is not None and date_val is not None:
                                # Normalize date to YYYY-MM-DD format for comparison
                                date_str = None
                                if isinstance(date_val, (date, datetime)):
                                    date_str = date_val.strftime("%Y-%m-%d")
                                elif isinstance(date_val, str):
                                    # Try to parse and normalize date string
                                    try:
                                        # Try common date formats
                                        for fmt in [
                                            "%Y-%m-%d",
                                            "%m/%d/%Y",
                                            "%Y/%m/%d",
                                            "%d/%m/%Y",
                                        ]:
                                            try:
                                                parsed = datetime.strptime(
                                                    date_val[:10], fmt
                                                )
                                                date_str = parsed.strftime("%Y-%m-%d")
                                                break
                                            except ValueError:
                                                continue
                                        if not date_str:
                                            # Fallback: take first 10 chars if it looks like a date
                                            if (
                                                len(date_val) >= 10
                                                and date_val[4] in ("-", "/")
                                                and date_val[7] in ("-", "/")
                                            ):
                                                date_str = date_val[:10].replace(
                                                    "/", "-"
                                                )
                                    except Exception:
                                        pass

                                if date_str:
                                    key = (str(sku_val).strip(), date_str)
                                    if key in existing_keys:
                                        is_duplicate = True

                    elif table_lower in ("slh", "sll"):
                        # Check listnum
                        listnum_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "listnum",
                                "list_num",
                                "list_number",
                                "list",
                            ):
                                listnum_idx = i
                                break
                        if listnum_idx is not None and listnum_idx < len(row):
                            listnum_val = row[listnum_idx]
                            if listnum_val is not None:
                                key = (str(listnum_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "cnt":
                        # Check code
                        code_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("code", "code_id", "cnt_code"):
                                code_idx = i
                                break
                        if code_idx is not None and code_idx < len(row):
                            code_val = row[code_idx]
                            if code_val is not None:
                                key = (str(code_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "cus":
                        # Check customer
                        customer_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "customer",
                                "cust",
                                "customer_id",
                                "cus_id",
                            ):
                                customer_idx = i
                                break
                        if customer_idx is not None and customer_idx < len(row):
                            customer_val = row[customer_idx]
                            if customer_val is not None:
                                key = (str(customer_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "vnd":
                        # Check vendor + vcode combination
                        vendor_idx = None
                        vcode_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("vendor", "vnd", "vendor_id", "vnd_id"):
                                vendor_idx = i
                            elif col.lower() in ("vcode", "vendor_code", "vnd_code"):
                                vcode_idx = i
                        if (
                            vendor_idx is not None
                            and vcode_idx is not None
                            and vendor_idx < len(row)
                            and vcode_idx < len(row)
                        ):
                            vendor_val = row[vendor_idx]
                            vcode_val = row[vcode_idx]
                            if vendor_val is not None and vcode_val is not None:
                                key = (str(vendor_val).strip(), str(vcode_val).strip())
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "str":
                        # Check store
                        store_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "store",
                                "store_id",
                                "store_num",
                                "store_number",
                                "str_store",
                            ):
                                store_idx = i
                                break
                        if store_idx is not None and store_idx < len(row):
                            store_val = row[store_idx]
                            if store_val is not None:
                                key = (str(store_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "cat":
                        # Check cat
                        cat_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in (
                                "cat",
                                "category",
                                "cat_id",
                                "category_id",
                            ):
                                cat_idx = i
                                break
                        if cat_idx is not None and cat_idx < len(row):
                            cat_val = row[cat_idx]
                            if cat_val is not None:
                                key = (str(cat_val).strip(),)
                                if key in existing_keys:
                                    is_duplicate = True

                    elif table_lower == "emp":
                        # Check id + uid combination
                        id_idx = None
                        uid_idx = None
                        for i, col in enumerate(col_names):
                            if col.lower() in ("id", "emp_id", "employee_id", "eid"):
                                id_idx = i
                            elif col.lower() in ("uid", "user_id", "userid", "emp_uid"):
                                uid_idx = i
                        if (
                            id_idx is not None
                            and uid_idx is not None
                            and id_idx < len(row)
                            and uid_idx < len(row)
                        ):
                            id_val = row[id_idx]
                            uid_val = row[uid_idx]
                            if id_val is not None and uid_val is not None:
                                key = (str(id_val).strip(), str(uid_val).strip())
                                if key in existing_keys:
                                    is_duplicate = True
                except (IndexError, TypeError, AttributeError):
                    # If we can't check, allow the insert (better to insert than skip incorrectly)
                    pass

                if is_duplicate:
                    skipped_duplicates += 1
                    continue

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
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid date" in error_msg or (
            "date" in error_msg and ("b'\\x00" in str(e) or "\\x00" in str(e))
        ):
            # This shouldn't happen if our error handling is working, but catch it anyway
            log_to_gui(
                f"WARNING: Date field error during row iteration: {e}. Some rows may have been skipped."
            )
            # Commit what we have so far
            if buf:
                try:
                    cur.executemany(insert_sql, buf)
                    inserted += len(buf)
                    batches += 1
                except Exception:
                    pass
        else:
            raise

    conn.commit()

    # Log duplicate skipping summary
    if duplicate_check_enabled and skipped_duplicates > 0:
        log_to_gui(f"Skipped {skipped_duplicates} duplicate record(s) in {table}")

    return inserted, batches


# ---------- Headless mode ----------

# Global reference to GUI log function (set by GUI mode)
_gui_log_func = None
_gui_progress_func = None
_gui_root = None


def log_to_gui(msg: str):
    """Route log message to GUI if available, otherwise use print."""
    global _gui_log_func
    logger.info(msg)
    if _gui_log_func:
        try:
            _gui_log_func(msg)
        except Exception:
            logger.debug("GUI log function failed; falling back to logger only.")


def update_gui_progress(percent: float, status: str):
    """Route progress update to GUI if available."""
    global _gui_progress_func, _gui_root
    if _gui_progress_func and _gui_root:
        try:
            _gui_root.after(0, lambda: _gui_progress_func(percent, status))
        except Exception:
            pass


def run_headless(
    cfg_path: Optional[str] = None,
    profile: Optional[str] = None,
    auto_sync: bool = False,
):
    """
    Run headless sync operation.

    Delta Sync Behavior:
    - When auto_sync=True, delta sync is automatically enabled if configured
    - Delta sync checks timestamp field in each table and only uploads newer records
    - Delta sync is automatically disabled when drop_recreate=True
    - Delta sync is automatically disabled when truncate_before_load=True (during truncate operation)
    - Date range filter works independently and can be combined with delta sync
    - When both delta sync and date range filter are enabled, uses max(since_date, date_range_start) as lower bound

    Date Field Validation:
    - If specified date_field is not found in DBF, a warning is issued and date filtering is disabled
    - Available fields are shown in the warning message for debugging
    - The date_field should be a timestamp field (e.g., 'tstamp', 'date', 'updated', 'created')
    """
    configure_logging()

    raw_cfg = load_config(cfg_path) if cfg_path else load_config()
    cfg = resolve_profile(raw_cfg, profile)

    engine = cfg["rds"].get("engine", "mysql")
    server = cfg["rds"]["server"]
    port = int(cfg["rds"].get("port", 3306 if engine == "mysql" else 1433))
    database = cfg["rds"]["database"]
    username = cfg["rds"]["username"]
    password = ensure_password(cfg)
    schema = cfg["rds"].get("schema") if engine == "mssql" else None

    # Admin API override
    admin_cfg = cfg.get("admin_api", {})
    if admin_cfg.get("enabled"):
        creds = fetch_admin_creds(
            admin_cfg.get("base_url", ""),
            admin_cfg.get("api_key", ""),
            str(admin_cfg.get("store_id", "")),
        )
        engine = creds.get("engine", engine)
        server = creds["host"]
        port = int(creds.get("port", port))
        database = creds["database"]
        username = creds["username"]
        password = creds["password"]
        schema = creds.get("schema") if engine == "mssql" else None

    folder = cfg["source"]["folder"]
    include = cfg["source"].get("include")

    # Copy files to sync directory (vfptordssync) to avoid interfering with source software
    log_to_gui(f"Copying DBF files to sync directory...")
    sync_dir, files = sync_files_to_directory(folder, include)
    log_to_gui(f"Files copied to sync directory: {sync_dir}")
    log_to_gui(
        f"Using copied files for sync operations (original files remain untouched)"
    )

    drop_recreate = bool(cfg["load"].get("drop_recreate", False))
    trunc = bool(cfg["load"].get("truncate_before_load", False))
    batch = int(cfg["load"].get("batch_size", 1000))
    tpref = cfg["load"].get("table_prefix", "")
    tsuff = cfg["load"].get("table_suffix", "")
    lower = bool(cfg["load"].get("coerce_lowercase_names", True))

    # Delta sync settings
    delta_cfg = cfg.get("delta_sync", {})
    # When auto_sync=True, force enable delta sync if configured (even if disabled in config, auto-sync implies delta)
    if auto_sync:
        delta_enabled = bool(delta_cfg.get("enabled", False)) and not drop_recreate
        if not delta_enabled:
            log_to_gui(
                "NOTE: Auto-sync is running but delta_sync.enabled is false. Running full sync."
            )
    else:
        delta_enabled = bool(delta_cfg.get("enabled", False)) and not drop_recreate

    date_field = delta_cfg.get(
        "date_field"
    )  # Default timestamp field to check (e.g., 'tstamp', 'date', 'updated', 'created')
    date_fields_per_table = delta_cfg.get(
        "date_fields_per_table", {}
    )  # Per-table date field overrides (e.g., {'inv': 'cdate', 'stk': 'updated'})
    related_table_date_fields = delta_cfg.get(
        "related_table_date_fields", {}
    )  # Per-table related table config (e.g., {'jnl': {'related_table': 'jnh', ...}})

    # Date range filter settings (for initial loads)
    date_range_cfg = cfg.get("load", {}).get("date_range_filter", {})
    date_range_enabled = bool(date_range_cfg.get("enabled", False))
    date_range_field = (
        date_range_cfg.get("date_field") or date_field
    )  # Fallback to delta_sync date_field
    date_range_start = None
    date_range_end = None

    if date_range_enabled and date_range_field:
        start_str = date_range_cfg.get("start_date")
        end_str = date_range_cfg.get("end_date")

        if start_str and start_str.strip():
            try:
                date_range_start = datetime.strptime(start_str.strip(), "%Y-%m-%d")
                log_to_gui(f"Date range filter: start_date = {date_range_start.date()}")
            except ValueError:
                log_to_gui(
                    f"WARNING: Invalid start_date format '{start_str}'. Use YYYY-MM-DD. Ignoring date range start."
                )

        if end_str and end_str.strip():
            try:
                date_range_end = datetime.strptime(end_str.strip(), "%Y-%m-%d")
                # Set to end of day for inclusive end date
                date_range_end = datetime.combine(date_range_end, datetime.max.time())
                log_to_gui(f"Date range filter: end_date = {date_range_end.date()}")
            except ValueError:
                log_to_gui(
                    f"WARNING: Invalid end_date format '{end_str}'. Use YYYY-MM-DD. Ignoring date range end."
                )

        if date_range_start or date_range_end:
            log_to_gui(f"Date range filter enabled: using field '{date_range_field}'")
    elif date_range_enabled and not date_range_field:
        log_to_gui(
            "WARNING: Date range filter enabled but no date_field specified. Disabling date range filter."
        )
        date_range_enabled = False

    # Always load sync tracking to track last sync time
    sync_tracking = load_sync_tracking()

    # Create tracking key from profile + (store_id when admin) + database + folder
    profile_key = profile or "default"
    if admin_cfg.get("enabled") and str(admin_cfg.get("store_id", "")).strip():
        tracking_key = f"{profile_key}|{admin_cfg.get('store_id')}|{database}|{folder}"
    else:
        tracking_key = f"{profile_key}|{database}|{folder}"

    # Warn if delta enabled but no date field specified
    if delta_enabled and not date_field:
        log_to_gui(
            "WARNING: Delta sync enabled but no date_field specified. Falling back to full sync."
        )
        delta_enabled = False

    # set global lowercase rule
    global safe_sql_name
    _orig_safe = safe_sql_name

    def _lower_override(name, coerce_lower=lower):
        return _orig_safe(name, coerce_lower=coerce_lower)

    safe_sql_name = _lower_override

    if engine == "mssql":
        conn = connect_mssql(server, database, username, password, port)
    else:
        conn = connect_mysql(server, database, username, password, port)

    files = list_allowed_dbfs(folder, include)

    log_to_gui(f"Found {len(files)} DBFs to load from {folder}")
    update_gui_progress(5, f"Found {len(files)} DBF file(s) to process...")

    # Log delta sync configuration
    if delta_enabled:
        if auto_sync:
            log_to_gui(
                f"Auto-sync with delta sync enabled: checking '{date_field}' field for records newer than last sync time"
            )
        else:
            log_to_gui(
                f"Delta sync enabled: using field '{date_field}' for incremental updates"
            )
        if trunc:
            log_to_gui(
                "  NOTE: Truncate is enabled - delta sync will be disabled during truncate operation"
            )

    # Log date range filter configuration
    if date_range_enabled and (date_range_start or date_range_end):
        log_to_gui(f"Date range filter enabled: using field '{date_range_field}'")
        if date_range_start:
            log_to_gui(f"  Start date: {date_range_start.date()}")
        if date_range_end:
            log_to_gui(f"  End date: {date_range_end.date()}")

    total_rows = 0
    sync_updates = {}  # Track new sync times per table
    total_files = len(files)

    update_gui_progress(10, f"Starting sync of {total_files} table(s)...")

    for file_idx, p in enumerate(files):
        # Update progress: 10% base + 80% for files (0-80%) + 10% reserved for completion
        progress_pct = (
            10 + int((file_idx / total_files) * 80) if total_files > 0 else 10
        )
        base = os.path.splitext(os.path.basename(p))[0]
        tgt = f"{tpref}{base}{tsuff}"
        table_key = f"{tracking_key}|{tgt}"

        # Update progress for this file
        update_gui_progress(progress_pct, f"Processing {base}...")

        # Determine date field for this specific table
        # Check per-table override first, then fall back to default
        table_date_field = date_fields_per_table.get(base.lower(), date_field)
        if not table_date_field:
            table_date_field = date_fields_per_table.get(base, date_field)

        # Check for related table date field config (e.g., jnl using jnh.tstamp, sll using slh.tstamp)
        related_table_config = related_table_date_fields.get(
            base.lower()
        ) or related_table_date_fields.get(base)
        if related_table_config:
            log_to_gui(
                f"  [Delta Sync] {tgt}: Using related table '{related_table_config.get('related_table')}' "
                f"field '{related_table_config.get('date_field_related')}' via join on '{related_table_config.get('join_field_local')}'"
            )

        try:
            since_date = None
            if (
                delta_enabled and not trunc
            ):  # Delta sync disabled when truncate is enabled
                # Get last sync time for this table
                last_sync_str = sync_tracking.get(table_key)
                if last_sync_str:
                    try:
                        since_date = datetime.fromisoformat(last_sync_str)
                        # Normalize timezone for logging
                        if since_date.tzinfo:
                            since_date_display = since_date.replace(tzinfo=None)
                        else:
                            since_date_display = since_date
                        if auto_sync:
                            log_to_gui(
                                f"  [Auto-Sync] {tgt}: Checking '{table_date_field or date_field}' field - uploading records newer than {since_date_display}"
                            )
                        else:
                            log_to_gui(
                                f"  [Delta Sync] {tgt}: Syncing records since {since_date_display} (using field '{table_date_field or date_field}')"
                            )
                    except Exception as e:
                        log_to_gui(
                            f"  [Delta Sync] WARNING: Could not parse last sync time for {tgt}: {e}. Doing full sync."
                        )
                        since_date = None
                else:
                    if auto_sync:
                        log_to_gui(
                            f"  [Auto-Sync] {tgt}: No previous sync found, doing initial full sync (all records)"
                        )
                    else:
                        log_to_gui(
                            f"  [Delta Sync] {tgt}: No previous sync found, doing full sync (all records)"
                        )
            elif trunc and delta_enabled:
                log_to_gui(
                    f"  [Delta Sync] {tgt}: Delta sync disabled due to truncate_before_load=True"
                )

            # Determine which date field to use for this table
            # Priority: date_range_field (if enabled) > table_date_field (per-table override) > date_field (default delta sync)
            if (
                date_range_enabled
                and date_range_field
                and (date_range_start or date_range_end)
            ):
                effective_date_field = date_range_field
            elif delta_enabled and table_date_field:
                effective_date_field = table_date_field
            elif delta_enabled and date_field:
                effective_date_field = date_field
            else:
                effective_date_field = None

            # Log per-table date field if different from default
            if delta_enabled and table_date_field and table_date_field != date_field:
                log_to_gui(
                    f"  [Delta Sync] {tgt}: Using table-specific date field '{table_date_field}' (default: '{date_field}')"
                )

            # Check if date field exists in this table (will be checked in iter_dbf_rows)
            # Note: If date field doesn't exist, iter_dbf_rows will warn and upload entire table

            # Only apply date range filter if enabled and we have at least one date bound
            effective_date_range_start = (
                date_range_start if date_range_enabled else None
            )
            effective_date_range_end = date_range_end if date_range_enabled else None

            # Validate date range if both bounds are provided
            if effective_date_range_start and effective_date_range_end:
                if effective_date_range_start > effective_date_range_end:
                    log_to_gui(
                        f"WARNING: {tgt} - start_date > end_date, swapping dates"
                    )
                    effective_date_range_start, effective_date_range_end = (
                        effective_date_range_end,
                        effective_date_range_start,
                    )

            # When drop_recreate is True, disable delta sync (since_date should be None)
            # When trunc is True, also disable delta sync for the truncate operation
            if drop_recreate:
                effective_since_date = None
                if delta_enabled:
                    log_to_gui(
                        f"  [Delta Sync] {tgt}: Delta sync disabled due to drop_recreate=True"
                    )
            elif trunc:
                # Delta sync will be disabled during truncate operation
                effective_since_date = None
            else:
                effective_since_date = since_date

            # Check if date field exists in DBF before upload (for logging)
            # This check happens inside bulk_insert/iter_dbf_rows, but we can pre-check
            dbf_preview = open_dbf(p)
            dbf_field_names = [f.name.lower() for f in dbf_preview.fields]
            date_field_exists = (
                effective_date_field and effective_date_field.lower() in dbf_field_names
            )

            if effective_date_field and not date_field_exists:
                log_to_gui(
                    f"  [Date Filter] {tgt}: Date field '{effective_date_field}' not found - uploading entire table"
                )

            if drop_recreate:
                # Drop & recreate fresh table, then load
                inserted, batches = bulk_insert(
                    conn,
                    engine,
                    tgt,
                    p,
                    batch,
                    schema,
                    recreate=True,
                    date_field=effective_date_field,
                    since_date=effective_since_date,
                    date_range_start=effective_date_range_start,
                    date_range_end=effective_date_range_end,
                    related_table_config=(
                        related_table_config
                        if effective_since_date or effective_date_range_start
                        else None
                    ),
                )
            else:
                # Reconcile schema; optional truncate then load
                inserted, batches = bulk_insert(
                    conn,
                    engine,
                    tgt,
                    p,
                    batch,
                    schema,
                    recreate=False,
                    date_field=effective_date_field,
                    since_date=effective_since_date,
                    date_range_start=effective_date_range_start,
                    date_range_end=effective_date_range_end,
                    related_table_config=(
                        related_table_config
                        if effective_since_date or effective_date_range_start
                        else None
                    ),
                )
                if trunc:
                    truncate_table(conn, engine, tgt, schema)
                    log_to_gui(
                        f"  [Truncate] {tgt}: Table truncated. Delta sync disabled for this load."
                    )
                    # When truncating, disable delta sync but keep date range filter if enabled
                    # This ensures we load all records within the date range after truncating
                    inserted, batches = bulk_insert(
                        conn,
                        engine,
                        tgt,
                        p,
                        batch,
                        schema,
                        recreate=False,
                        date_field=effective_date_field,
                        since_date=None,  # Explicitly disable delta sync when truncating
                        date_range_start=effective_date_range_start,
                        date_range_end=effective_date_range_end,
                        related_table_config=None,  # Disable related table filtering when truncating
                    )

            total_rows += inserted

            # Update progress after file completion
            file_progress = (
                10 + int(((file_idx + 1) / total_files) * 80) if total_files > 0 else 90
            )
            update_gui_progress(
                file_progress, f"Completed {base} - {inserted} rows inserted"
            )

            # Update sync tracking (always update after successful sync)
            # This timestamp is used for the next delta sync to check for newer records
            if inserted >= 0:  # Update even if 0 rows (indicates successful sync)
                # Store ISO with local timezone info - this becomes the "since_date" for next sync
                sync_timestamp = datetime.now().astimezone().isoformat()
                sync_updates[table_key] = sync_timestamp
                if delta_enabled and not drop_recreate and not trunc:
                    if auto_sync:
                        log_to_gui(
                            f"  [Auto-Sync] {tgt}: Sync complete. Next sync will check for records newer than {sync_timestamp}"
                        )
                    else:
                        log_to_gui(
                            f"  [Delta Sync] {tgt}: Sync timestamp updated to {sync_timestamp}"
                        )

            # Determine sync mode for logging
            if drop_recreate:
                sync_mode = "full (drop_recreate)"
            elif trunc:
                sync_mode = "full (truncated)"
            elif delta_enabled and since_date:
                sync_mode = f"delta (since {since_date.replace(tzinfo=None) if since_date and since_date.tzinfo else since_date})"
            elif date_range_enabled and (
                effective_date_range_start or effective_date_range_end
            ):
                range_desc = []
                if effective_date_range_start:
                    range_desc.append(f"from {effective_date_range_start.date()}")
                if effective_date_range_end:
                    range_desc.append(f"to {effective_date_range_end.date()}")
                sync_mode = f"date_range ({' '.join(range_desc)})"
            else:
                sync_mode = "full"

            log_to_gui(
                f"Loaded {inserted:>8} rows → {tgt} ({batches} batch/es) [{sync_mode}]"
            )
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error message for common date-related errors
            if "invalid date" in error_msg.lower() or (
                "date" in error_msg.lower()
                and ("b'\\x00" in error_msg or "\\x00" in error_msg)
            ):
                log_to_gui(f"WARNING {base}: Invalid date detected - {error_msg}")
                log_to_gui(
                    f"  → This may be due to uninitialized/null date values in the DBF file."
                )
                log_to_gui(
                    f"  → Rows with invalid dates are automatically skipped during filtering."
                )
                log_to_gui(f"  → Continuing with remaining rows...")
                # Don't treat this as a fatal error - continue processing
                continue
            else:
                log_to_gui(f"ERROR {base}: {e}")

    # Save updated sync tracking (always save to track last sync time)
    if sync_updates:
        sync_tracking.update(sync_updates)
        # Update last auto-sync time if this was an auto-sync run
        if auto_sync:
            sync_tracking["__last_auto_sync__"] = (
                datetime.now().astimezone().isoformat()
            )
        save_sync_tracking(sync_tracking)
        if delta_enabled:
            if auto_sync:
                log_to_gui(
                    f"\n[Auto-Sync] Sync complete. Updated sync tracking for {len(sync_updates)} table(s)"
                )
            else:
                log_to_gui(
                    f"\n[Delta Sync] Updated sync tracking for {len(sync_updates)} table(s)"
                )

    try:
        conn.close()
    except Exception:
        pass

    update_gui_progress(100, f"Complete! Inserted {total_rows} rows total.")
    log_to_gui(f"ALL DONE. Inserted {total_rows} rows total.")


def cli_init(path: Optional[str] = None):
    print("\n=== Initial Setup (creates a saved config) ===")
    engine = input("Engine (mysql/mssql) [mysql]: ").strip().lower() or "mysql"
    server = input("RDS server/host: ").strip()
    port = input("Port [3306 for mysql, 1433 for mssql]: ").strip()
    port = int(port) if port else (3306 if engine == "mysql" else 1433)
    database = input("Database name: ").strip()
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    folder = input(r"Path to ksv\data folder (e.g., C:/ksv/data): ").strip()
    schema = input("Schema (mssql) [dbo]: ").strip() if engine == "mssql" else None
    schema = schema or ("dbo" if engine == "mssql" else None)

    cfg = {
        "rds": {
            "engine": engine,
            "server": server,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "schema": schema,
        },
        "source": {"folder": folder},
        "load": {
            "drop_recreate": True,
            "truncate_before_load": False,
            "batch_size": 1000,
            "table_prefix": "",
            "table_suffix": "",
            "coerce_lowercase_names": True,
        },
    }
    cfg_path = path or default_config_path()
    save_config(cfg, cfg_path)
    print(f"Saved config to {cfg_path}.")


# ---------- DearPyGui GUI (kept, optional) ----------


def run_gui():
    configure_logging()

    try:
        import dearpygui.dearpygui as dpg
    except Exception:
        logger.error("DearPyGui not available. Install with: pip install dearpygui")
        sys.exit(1)

    dpg.create_context()
    state = {"files": [], "folder": ""}

    def log(msg: str):
        old = dpg.get_value("log_box") if dpg.does_item_exist("log_box") else ""
        dpg.set_value("log_box", (old + msg + "\n")[-20000:])

    def scan_dbfs():
        folder = dpg.get_value("folder_input")
        if not folder or not os.path.isdir(folder):
            log("Please select a valid folder.")
            return
        files = list_allowed_dbfs(folder)
        state["folder"] = folder
        state["files"] = files
        dpg.configure_item("files_list", items=[os.path.basename(f) for f in files])
        dpg.set_value("count_text", f"Found: {len(files)}")
        log(f"Scanned {len(files)} DBFs.")

    def choose_folder(sender, app_data):
        path = app_data.get("file_path_name") or app_data.get("current_path")
        if path and os.path.isdir(path):
            dpg.set_value("folder_input", path)
            scan_dbfs()

    def save_cfg():
        engine = dpg.get_value("engine_combo")
        server = dpg.get_value("server_input")
        port = int(dpg.get_value("port_input") or (3306 if engine == "mysql" else 1433))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input") or ""
        schema = dpg.get_value("schema_input") if engine == "mssql" else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ""
        tsuff = dpg.get_value("tsuff_input") or ""

        # Persist current settings (including password) to config
        cfg = {
            "rds": {
                "engine": engine,
                "server": server,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "schema": schema,
            },
            "source": {"folder": state.get("folder", "")},
            "load": {
                "drop_recreate": recreate,
                "truncate_before_load": trunc,
                "batch_size": batch,
                "table_prefix": tpref,
                "table_suffix": tsuff,
                "coerce_lowercase_names": True,
            },
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
        port = int(dpg.get_value("port_input") or (3306 if engine == "mysql" else 1433))
        database = dpg.get_value("db_input")
        username = dpg.get_value("user_input")
        password = dpg.get_value("pwd_input")
        schema = dpg.get_value("schema_input") if engine == "mssql" else None
        recreate = bool(dpg.get_value("recreate_chk"))
        trunc = bool(dpg.get_value("trunc_chk"))
        batch = int(dpg.get_value("batch_input") or 1000)
        tpref = dpg.get_value("tpref_input") or ""
        tsuff = dpg.get_value("tsuff_input") or ""

        # Persist current settings (including password) to config
        cfg = {
            "rds": {
                "engine": engine,
                "server": server,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "schema": schema,
            },
            "source": {"folder": state.get("folder", "")},
            "load": {
                "drop_recreate": recreate,
                "truncate_before_load": trunc,
                "batch_size": batch,
                "table_prefix": tpref,
                "table_suffix": tsuff,
                "coerce_lowercase_names": True,
            },
        }
        save_config(cfg, default_config_path())

        try:
            if engine == "mssql":
                conn = connect_mssql(server, database, username, password, port)
            else:
                conn = connect_mysql(server, database, username, password, port)
        except Exception as e:
            log(f"Connection failed: {e}")
            return

        total_rows = 0
        try:
            for idx in sel_indices:
                p = state["files"][idx]
                base = os.path.splitext(os.path.basename(p))[0]
                tgt = f"{tpref}{base}{tsuff}"
                try:
                    if recreate:
                        inserted, batches = bulk_insert(
                            conn, engine, tgt, p, batch, schema, recreate=True
                        )
                    else:
                        inserted, batches = bulk_insert(
                            conn, engine, tgt, p, batch, schema, recreate=False
                        )
                        if trunc:
                            truncate_table(conn, engine, tgt, schema)
                            inserted, batches = bulk_insert(
                                conn, engine, tgt, p, batch, schema, recreate=False
                            )
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
    with dpg.window(
        label="VFP DBF → RDS Uploader (DearPyGui)", width=980, height=700, pos=(50, 50)
    ):
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="ksv/data folder", tag="folder_input", width=600)
            dpg.add_button(
                label="Browse", callback=lambda: dpg.show_item("folder_dialog")
            )
            dpg.add_button(label="Scan DBFs", callback=scan_dbfs)
            dpg.add_text("Found: 0", tag="count_text")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_listbox(
                items=[], tag="files_list", width=400, num_items=16, label="DBF Tables"
            )
            with dpg.group():
                dpg.add_combo(
                    ["mysql", "mssql"],
                    default_value="mysql",
                    label="Engine",
                    tag="engine_combo",
                )
                dpg.add_input_text(label="Server/Host", tag="server_input")
                dpg.add_input_text(label="Port", tag="port_input", default_value="3306")
                dpg.add_input_text(label="Database", tag="db_input")
                dpg.add_input_text(label="Username", tag="user_input")
                dpg.add_input_text(label="Password", tag="pwd_input", password=True)
                dpg.add_input_text(
                    label="Schema (mssql)", tag="schema_input", default_value="dbo"
                )
                dpg.add_checkbox(
                    label="Drop & recreate tables",
                    tag="recreate_chk",
                    default_value=True,
                )
                dpg.add_checkbox(label="Truncate before load", tag="trunc_chk")
                dpg.add_input_text(
                    label="Batch size", tag="batch_input", default_value="1000"
                )
                dpg.add_input_text(label="Table prefix", tag="tpref_input")
                dpg.add_input_text(label="Table suffix", tag="tsuff_input")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Config", callback=save_cfg)
                    dpg.add_button(label="Upload Selected", callback=upload_selected)
        dpg.add_separator()
        dpg.add_input_text(
            tag="log_box", multiline=True, readonly=True, height=220, width=940
        )

    with dpg.file_dialog(
        directory_selector=True, show=False, callback=choose_folder, tag="folder_dialog"
    ):
        dpg.add_file_extension(".dbf")

    dpg.create_viewport(title="DBF → RDS Uploader", width=1100, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# ---------- Single Instance Detection ----------


def check_single_instance():
    """Check if another instance is already running. Returns (is_first_instance, mutex_handle)."""
    try:
        import win32event
        import win32api
        import winerror

        # Create a named mutex for single instance detection
        mutex_name = "VFP_DBF_Uploader_SingleInstance"
        mutex = win32event.CreateMutex(None, False, mutex_name)
        last_error = win32api.GetLastError()

        if last_error == winerror.ERROR_ALREADY_EXISTS:
            # Another instance is running
            return False, None
        return True, mutex
    except ImportError:
        # Fallback: use lock file (works on all platforms)
        import tempfile

        lock_file = os.path.join(tempfile.gettempdir(), "vfp_dbf_uploader.lock")

        try:
            # Try to create lock file exclusively
            if os.path.exists(lock_file):
                # Check if process is still running
                try:
                    with open(lock_file, "r") as f:
                        pid = int(f.read().strip())
                    # Check if process exists (Windows)
                    try:
                        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
                        return False, None  # Process exists
                    except (OSError, ProcessLookupError):
                        # Process doesn't exist, remove stale lock
                        os.remove(lock_file)
                except (ValueError, IOError):
                    os.remove(lock_file)

            # Create new lock file
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
            return True, lock_file
        except Exception:
            return True, None  # Allow if we can't create lock


def cleanup_instance_lock(lock_handle):
    """Clean up the instance lock."""
    if lock_handle:
        try:
            import win32api

            win32api.CloseHandle(lock_handle)
        except (ImportError, Exception):
            # Lock file cleanup
            try:
                if isinstance(lock_handle, str) and os.path.exists(lock_handle):
                    os.remove(lock_handle)
            except Exception:
                pass


def bring_window_to_front():
    """Try to bring existing window to front (Windows)."""
    try:
        import win32gui
        import win32con

        def enum_handler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if "VFP DBF → RDS Uploader" in window_title:
                    # Restore if minimized
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    # Bring to front
                    win32gui.SetForegroundWindow(hwnd)
                    win32gui.BringWindowToTop(hwnd)
                    return False  # Stop enumeration
            return True

        win32gui.EnumWindows(enum_handler, None)
        return True
    except ImportError:
        return False


# ---------- Tkinter GUI (Modernized) ----------


def run_gui_tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    configure_logging()

    # Add error logging to file for debugging when running as exe
    error_log_path = None
    try:
        # Try to create error log in temp directory or next to exe
        if getattr(sys, "frozen", False):
            # Running as compiled exe
            error_log_path = os.path.join(
                os.path.dirname(sys.executable), "vfp_uploader_error.log"
            )
        else:
            # Running as script
            error_log_path = os.path.join(
                os.path.dirname(__file__), "vfp_uploader_error.log"
            )
    except Exception:
        error_log_path = None

    def log_error(msg):
        """Log error to file and console"""
        print(f"ERROR: {msg}", file=sys.stderr)
        if error_log_path:
            try:
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()}: {msg}\n")
            except Exception:
                pass

    # Check for single instance BEFORE creating any GUI elements
    try:
        is_first_instance, lock_handle = check_single_instance()
        if not is_first_instance:
            # Another instance is running - try to bring it to front
            log_error(
                "Another instance is already running. Attempting to bring it to front..."
            )
            if bring_window_to_front():
                log_error("Existing window brought to front. Exiting this instance.")
                return  # Existing window brought to front, exit this instance
            else:
                # Couldn't bring to front - show message if possible, otherwise just exit
                try:
                    # Create a minimal Tk root just for the message box
                    temp_root = tk.Tk()
                    temp_root.withdraw()  # Hide the root window
                    messagebox.showinfo(
                        "Already Running",
                        "Another instance is already running.\nCheck the system tray or taskbar.",
                    )
                    temp_root.destroy()
                except Exception as e:
                    log_error(f"Could not show message box: {e}")
                log_error("Exiting duplicate instance.")
                return
    except Exception as e:
        log_error(f"Error in single instance check: {e}")
        # For now, continue - might be a pywin32 issue, but log it
        lock_handle = None
        # Don't exit - allow it to continue but log the issue

    try:
        # Resolve or ask for a data folder once; remember it next to the script
        ksv = find_ksv_root()
        if not ksv:
            try:
                messagebox.showinfo(
                    "Select Data Folder",
                    "Please select the folder that contains your DBF files. We'll remember it.",
                )
                chosen = filedialog.askdirectory(
                    title="Select the folder that contains your DBF files"
                )
                if not chosen:
                    messagebox.showerror(
                        "Folder required", "You must select a folder to continue."
                    )
                    return
                script_dir = (
                    os.path.dirname(sys.executable)
                    if getattr(sys, "frozen", False)
                    else os.path.dirname(__file__)
                )
                hint_path = Path(script_dir) / "ksv_path.txt"
                hint_path.write_text(chosen, encoding="utf-8")
                ksv = chosen
            except Exception as e:
                log_error(f"Error selecting data folder: {e}")
                # Try a default
                ksv = "C:\\ksv"
                if not os.path.isdir(ksv):
                    log_error(f"Default data folder not found: {ksv}")
                    return
    except Exception as e:
        log_error(f"Error finding data folder: {e}")
        # Try a default
        ksv = "C:\\ksv"
        if not os.path.isdir(ksv):
            log_error(f"Default data folder not found: {ksv}")
            return

    state = {"files": [], "folder": ""}

    # Upload button reference (will be set after button creation)
    upload_btn_ref = {"btn": None}

    # Upload thread control (must be initialized before upload_selected is defined)
    _upload_thread = None
    _upload_stop = threading.Event()

    # Global log function (will be set after log_box is created)
    # Declare globals at the start of the function
    global _gui_log_func, _gui_progress_func, _gui_root

    def log(msg: str):
        """Thread-safe log function that writes to GUI log box."""
        if log_box:
            root.after(0, lambda m=msg: _log_to_box(m))
        else:
            # Fallback to console if log_box not ready
            print(msg)

    def _log_to_box(msg: str):
        """Internal function to write to log box (called from main thread)."""
        try:
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")
        except Exception:
            pass  # Ignore errors if log_box is destroyed

    # Store log and progress functions globally so headless mode can use them
    # We'll set them after log_box and update_progress are defined (see end of function)

    def scan_dbfs():
        folder = folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder", "Please select a valid folder.")
            return

        # Copy files to sync directory first
        log("Copying DBF files and related files (CDX, FPT) to sync directory...")
        try:
            sync_dir, copied_files = sync_files_to_directory(folder)
            log(f"Files copied to sync directory: {sync_dir}")
            log(f"Using copied files for operations (original files remain untouched)")

            # Use copied files from sync directory
            state["folder"] = sync_dir  # Store sync directory as the working folder
            state["source_folder"] = folder  # Keep track of original folder
            state["files"] = copied_files
            files_list.delete(0, "end")
            for f in copied_files:
                files_list.insert("end", os.path.basename(f))
            count_var.set(f"Found: {len(copied_files)}")
            log(f"Scanned {len(copied_files)} DBFs from sync directory.")
        except Exception as e:
            log(f"ERROR: Could not copy files to sync directory: {e}")
            messagebox.showerror(
                "Error", f"Could not copy files to sync directory: {e}"
            )
            # Fallback to original folder
            files = list_allowed_dbfs(folder)
            state["folder"] = folder
            state["files"] = files
            files_list.delete(0, "end")
            for f in files:
                files_list.insert("end", os.path.basename(f))
            count_var.set(f"Found: {len(files)}")
            log(f"Scanned {len(files)} DBFs (using original files).")

    def browse_folder():
        folder = filedialog.askdirectory(
            title="Select ksv/data folder", initialdir=os.path.join(ksv, "data")
        )
        if folder:
            folder_var.set(folder)
            scan_dbfs()

    def load_cfg():
        try:
            cfg = load_config(default_config_path())
            if "profiles" in cfg:
                prof = next(iter(cfg["profiles"].keys()))
                eff = cfg["profiles"][prof]
            else:
                eff = cfg
            r = eff["rds"]
            s = eff["source"]
            l = eff["load"]
            admin = eff.get("admin_api", {})
            engine_var.set(r.get("engine", "mysql"))
            server_var.set(r.get("server", ""))
            port_var.set(
                str(
                    r.get("port", 3306 if r.get("engine", "mysql") == "mysql" else 1433)
                )
            )
            db_var.set(r.get("database", ""))
            user_var.set(r.get("username", ""))
            schema_var.set(
                r.get("schema", "") if r.get("engine", "mysql") == "mssql" else ""
            )
            folder_var.set(s.get("folder", "") or os.path.join(ksv, "data"))
            trunc_var.set(1 if l.get("truncate_before_load") else 0)
            recreate_var.set(1 if l.get("drop_recreate", True) else 0)
            batch_var.set(str(l.get("batch_size", 1000)))
            tpref_var.set(l.get("table_prefix", ""))
            tsuff_var.set(l.get("table_suffix", ""))
            # Load password if available
            if "password" in r:
                pwd_var.set(r["password"])

            # Load delta sync settings (map None to empty string for UI)
            delta = eff.get("delta_sync", {})
            delta_enabled_var.set(1 if delta.get("enabled", False) else 0)
            _date_field_val = delta.get("date_field")
            delta_date_field_var.set(
                _date_field_val
                if isinstance(_date_field_val, str) and _date_field_val
                else ""
            )
            delta_interval_var.set(str(delta.get("auto_sync_interval_seconds", 3600)))

            # Load date range filter settings
            date_range = l.get("date_range_filter", {})
            date_range_enabled_var.set(1 if date_range.get("enabled", False) else 0)
            _date_range_field_val = date_range.get("date_field")
            date_range_field_var.set(
                _date_range_field_val
                if isinstance(_date_range_field_val, str) and _date_range_field_val
                else ""
            )
            _start_date_val = date_range.get("start_date")
            date_range_start_var.set(
                _start_date_val
                if isinstance(_start_date_val, str) and _start_date_val
                else ""
            )
            _end_date_val = date_range.get("end_date")
            date_range_end_var.set(
                _end_date_val
                if isinstance(_end_date_val, str) and _end_date_val
                else ""
            )

            # Admin API settings
            admin_enabled_var.set(1 if admin.get("enabled") else 0)
            admin_base_url_var.set(admin.get("base_url", ""))
            admin_api_key_var.set(admin.get("api_key", ""))
            admin_store_id_var.set(str(admin.get("store_id", "")))

            log("Loaded values from existing config.")
        except FileNotFoundError:
            messagebox.showinfo(
                "Load Config",
                "No config found yet. Fill the form and click Save Config.",
            )
        except Exception as e:
            messagebox.showerror("Load Config", str(e))

    def save_cfg():
        engine = engine_var.get()
        try:
            # Load existing config to preserve sections not in GUI (e.g., related_table_date_fields, date_fields_per_table)
            existing_cfg = {}
            try:
                existing_cfg = load_config(default_config_path())
            except (FileNotFoundError, Exception):
                pass  # No existing config, start fresh

            # Build new config from GUI values
            cfg = {
                "rds": {
                    "engine": engine,
                    "server": server_var.get().strip(),
                    "port": int(
                        port_var.get() or (3306 if engine == "mysql" else 1433)
                    ),
                    "database": db_var.get().strip(),
                    "username": user_var.get().strip(),
                    "password": pwd_var.get(),
                    "schema": schema_var.get().strip() if engine == "mssql" else None,
                },
                "source": {
                    "folder": folder_var.get().strip() or os.path.join(ksv, "data")
                },
                "load": {
                    "drop_recreate": bool(recreate_var.get()),
                    "truncate_before_load": bool(trunc_var.get()),
                    "batch_size": int(batch_var.get() or 1000),
                    "table_prefix": tpref_var.get().strip(),
                    "table_suffix": tsuff_var.get().strip(),
                    "coerce_lowercase_names": True,
                    "date_range_filter": {
                        "enabled": bool(date_range_enabled_var.get()),
                        "date_field": date_range_field_var.get().strip() or None,
                        "start_date": date_range_start_var.get().strip() or None,
                        "end_date": date_range_end_var.get().strip() or None,
                    },
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

            # Preserve sections that aren't in the GUI from existing config
            if existing_cfg:
                # Preserve date_fields_per_table if it exists
                if (
                    "delta_sync" in existing_cfg
                    and "date_fields_per_table" in existing_cfg.get("delta_sync", {})
                ):
                    if "date_fields_per_table" not in cfg["delta_sync"]:
                        cfg["delta_sync"]["date_fields_per_table"] = existing_cfg[
                            "delta_sync"
                        ]["date_fields_per_table"]

                # Preserve related_table_date_fields if it exists
                if (
                    "delta_sync" in existing_cfg
                    and "related_table_date_fields"
                    in existing_cfg.get("delta_sync", {})
                ):
                    if "related_table_date_fields" not in cfg["delta_sync"]:
                        cfg["delta_sync"]["related_table_date_fields"] = existing_cfg[
                            "delta_sync"
                        ]["related_table_date_fields"]

                # Preserve profiles if it exists (multi-profile configs)
                if "profiles" in existing_cfg:
                    cfg["profiles"] = existing_cfg["profiles"]
        except ValueError:
            messagebox.showerror(
                "Config", "Port, Batch size, and Auto-sync interval must be numbers."
            )
            return

        path = save_config(cfg, default_config_path())
        log(f"Saved config to {path}.")

    def update_progress(percent: float, status: str):
        """Thread-safe progress bar update."""
        progress_var.set(percent)
        progress_status_var.set(status)
        # Update percentage display
        progress_percent_var.set(f"{int(percent)}%")
        if percent > 0:
            progress_frame.grid()
        if percent >= 100:
            root.after(
                1000, lambda: progress_frame.grid_remove()
            )  # Hide after 1 second

    def upload_selected():
        sel = files_list.curselection()
        if not sel:
            messagebox.showinfo("Upload", "No DBFs selected.")
            return

        # Check if upload is already running
        nonlocal _upload_thread, _upload_stop
        if _upload_thread and _upload_thread.is_alive():
            messagebox.showinfo("Upload", "Upload already in progress. Please wait.")
            return

        # Disable upload button during operation
        if upload_btn_ref["btn"]:
            upload_btn_ref["btn"].config(state="disabled")
        progress_frame.grid()
        progress_var.set(0.0)
        progress_status_var.set("Starting upload...")
        _upload_stop.clear()

        def upload_worker():
            """Background worker for upload operation."""
            nonlocal _upload_stop
            try:
                engine = engine_var.get()
                server = server_var.get().strip()
                database = db_var.get().strip()
                username = user_var.get().strip()
                password = pwd_var.get()
                try:
                    port = int(port_var.get() or (3306 if engine == "mysql" else 1433))
                except ValueError:
                    root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Upload", "Port must be a number."
                        ),
                    )
                    root.after(
                        0,
                        lambda: (
                            upload_btn_ref["btn"].config(state="normal")
                            if upload_btn_ref["btn"]
                            else None
                        ),
                    )
                    root.after(0, lambda: progress_frame.grid_remove())
                    return
                schema = schema_var.get().strip() if engine == "mssql" else None

                trunc = bool(trunc_var.get())
                recreate = bool(recreate_var.get())
                try:
                    batch = int(batch_var.get() or 1000)
                except ValueError:
                    root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Upload", "Batch size must be a number."
                        ),
                    )
                    root.after(
                        0,
                        lambda: (
                            upload_btn_ref["btn"].config(state="normal")
                            if upload_btn_ref["btn"]
                            else None
                        ),
                    )
                    root.after(0, lambda: progress_frame.grid_remove())
                    return

                tpref = tpref_var.get().strip()
                tsuff = tsuff_var.get().strip()

                # Admin API override
                use_admin = bool(admin_enabled_var.get())
                if use_admin:
                    root.after(
                        0,
                        lambda: update_progress(
                            5, "Fetching credentials from admin backend..."
                        ),
                    )
                    try:
                        creds = fetch_admin_creds(
                            admin_base_url_var.get().strip(),
                            admin_api_key_var.get().strip(),
                            admin_store_id_var.get().strip(),
                        )
                        engine = creds.get("engine", engine)
                        server = creds["host"]
                        port = int(creds.get("port", port))
                        database = creds["database"]
                        username = creds["username"]
                        password = creds["password"]
                        schema = creds.get("schema") if engine == "mssql" else None
                        root.after(
                            0,
                            lambda: log(
                                "Fetched store credentials from Admin backend."
                            ),
                        )
                    except Exception as e:
                        root.after(
                            0,
                            lambda: messagebox.showerror(
                                "Admin creds", f"Failed to fetch creds: {e}"
                            ),
                        )
                        root.after(
                            0,
                            lambda: (
                                upload_btn_ref["btn"].config(state="normal")
                                if upload_btn_ref["btn"]
                                else None
                            ),
                        )
                        root.after(0, lambda: progress_frame.grid_remove())
                        return

                # Persist current settings to config (preserve manual sections)
                try:
                    # Load existing config to preserve sections not in GUI
                    existing_cfg = {}
                    try:
                        existing_cfg = load_config(default_config_path())
                    except (FileNotFoundError, Exception):
                        pass

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
                        "source": {
                            "folder": folder_var.get().strip()
                            or os.path.join(ksv, "data")
                        },
                        "load": {
                            "drop_recreate": bool(recreate),
                            "truncate_before_load": bool(trunc),
                            "batch_size": int(batch),
                            "table_prefix": tpref,
                            "table_suffix": tsuff,
                            "coerce_lowercase_names": True,
                            "date_range_filter": {
                                "enabled": bool(date_range_enabled_var.get()),
                                "date_field": date_range_field_var.get().strip()
                                or None,
                                "start_date": date_range_start_var.get().strip()
                                or None,
                                "end_date": date_range_end_var.get().strip() or None,
                            },
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
                            "auto_sync_interval_seconds": int(
                                delta_interval_var.get() or 3600
                            ),
                        },
                    }

                    # Preserve sections that aren't in the GUI from existing config
                    if existing_cfg:
                        # Preserve date_fields_per_table if it exists
                        if (
                            "delta_sync" in existing_cfg
                            and "date_fields_per_table"
                            in existing_cfg.get("delta_sync", {})
                        ):
                            if "date_fields_per_table" not in cfg_persist["delta_sync"]:
                                cfg_persist["delta_sync"]["date_fields_per_table"] = (
                                    existing_cfg["delta_sync"]["date_fields_per_table"]
                                )

                        # Preserve related_table_date_fields if it exists
                        if (
                            "delta_sync" in existing_cfg
                            and "related_table_date_fields"
                            in existing_cfg.get("delta_sync", {})
                        ):
                            if (
                                "related_table_date_fields"
                                not in cfg_persist["delta_sync"]
                            ):
                                cfg_persist["delta_sync"][
                                    "related_table_date_fields"
                                ] = existing_cfg["delta_sync"][
                                    "related_table_date_fields"
                                ]

                        # Preserve profiles if it exists (multi-profile configs)
                        if "profiles" in existing_cfg:
                            cfg_persist["profiles"] = existing_cfg["profiles"]

                    save_config(cfg_persist, default_config_path())
                except Exception:
                    pass

                root.after(0, lambda: update_progress(10, "Connecting to database..."))
                try:
                    if engine == "mssql":
                        conn = connect_mssql(server, database, username, password, port)
                    else:
                        conn = connect_mysql(server, database, username, password, port)
                except Exception as e:
                    root.after(
                        0, lambda: messagebox.showerror("Connection failed", str(e))
                    )
                    root.after(
                        0,
                        lambda: (
                            upload_btn_ref["btn"].config(state="normal")
                            if upload_btn_ref["btn"]
                            else None
                        ),
                    )
                    root.after(0, lambda: progress_frame.grid_remove())
                    return

                # Delta sync settings from UI
                delta_enabled = bool(delta_enabled_var.get()) and not recreate
                date_field_cfg = (delta_date_field_var.get() or "").strip()
                date_field = (
                    date_field_cfg if delta_enabled and date_field_cfg else None
                )

                # Date range filter settings from UI
                date_range_enabled = bool(date_range_enabled_var.get())
                date_range_field_cfg = (date_range_field_var.get() or "").strip()
                # Fallback to delta sync date_field if not specified
                if not date_range_field_cfg and date_field:
                    date_range_field_cfg = date_field
                date_range_field = (
                    date_range_field_cfg
                    if date_range_enabled and date_range_field_cfg
                    else None
                )
                date_range_start = None
                date_range_end = None

                if date_range_enabled and date_range_field:
                    start_str = date_range_start_var.get().strip()
                    end_str = date_range_end_var.get().strip()

                    if start_str:
                        try:
                            date_range_start = datetime.strptime(start_str, "%Y-%m-%d")
                            root.after(
                                0,
                                lambda s=start_str: log(
                                    f"Date range filter: start_date = {date_range_start.date()}"
                                ),
                            )
                        except ValueError:
                            root.after(
                                0,
                                lambda s=start_str: log(
                                    f"WARNING: Invalid start_date format '{s}'. Use YYYY-MM-DD. Ignoring date range start."
                                ),
                            )

                    if end_str:
                        try:
                            date_range_end = datetime.strptime(end_str, "%Y-%m-%d")
                            # Set to end of day for inclusive end date
                            date_range_end = datetime.combine(
                                date_range_end, datetime.max.time()
                            )
                            root.after(
                                0,
                                lambda: log(
                                    f"Date range filter: end_date = {date_range_end.date()}"
                                ),
                            )
                        except ValueError:
                            root.after(
                                0,
                                lambda s=end_str: log(
                                    f"WARNING: Invalid end_date format '{s}'. Use YYYY-MM-DD. Ignoring date range end."
                                ),
                            )

                    # Validate date range if both bounds are provided
                    if date_range_start and date_range_end:
                        if date_range_start > date_range_end:
                            root.after(
                                0,
                                lambda: log(
                                    f"WARNING: start_date > end_date, swapping dates"
                                ),
                            )
                            date_range_start, date_range_end = (
                                date_range_end,
                                date_range_start,
                            )

                if date_range_enabled and not date_range_field:
                    root.after(
                        0,
                        lambda: log(
                            "WARNING: Date range filter enabled but no date_field specified. Disabling date range filter."
                        ),
                    )
                    date_range_enabled = False

                # Load sync tracking for incremental sync
                sync_tracking = load_sync_tracking()
                profile_key = "default"
                folder = folder_var.get().strip() or os.path.join(ksv, "data")
                if use_admin and admin_store_id_var.get().strip():
                    tracking_key = f"{profile_key}|{admin_store_id_var.get().strip()}|{database}|{folder}"
                else:
                    tracking_key = f"{profile_key}|{database}|{folder}"

                total_rows = 0
                sync_updates = {}
                total_files = len(sel)

                root.after(
                    0,
                    lambda: update_progress(15, f"Processing {total_files} file(s)..."),
                )

                try:
                    for file_idx, idx in enumerate(sel):
                        if _upload_stop.is_set():
                            root.after(0, lambda: log("Upload cancelled by user."))
                            break

                        p = state["files"][idx]
                        base = os.path.splitext(os.path.basename(p))[0]
                        tgt = f"{tpref}{base}{tsuff}"
                        table_key = f"{tracking_key}|{tgt}"

                        # Update progress
                        progress_pct = 15 + int((file_idx / total_files) * 80)
                        root.after(
                            0,
                            lambda b=base, p=progress_pct: update_progress(
                                p, f"Processing {b}..."
                            ),
                        )

                        try:
                            # Determine since_date for delta sync
                            since_date = None
                            if date_field:
                                last_sync_str = sync_tracking.get(table_key)
                                if last_sync_str:
                                    try:
                                        since_date = datetime.fromisoformat(
                                            last_sync_str
                                        )
                                        root.after(
                                            0,
                                            lambda t=tgt, s=since_date: log(
                                                f"Delta: {t} since {s}"
                                            ),
                                        )
                                    except Exception:
                                        since_date = None

                            # Determine effective date field and date range
                            # Priority: date_range_field (if enabled) > date_field (delta sync)
                            if (
                                date_range_enabled
                                and date_range_field
                                and (date_range_start or date_range_end)
                            ):
                                effective_date_field = date_range_field
                            elif delta_enabled and date_field:
                                effective_date_field = date_field
                            else:
                                effective_date_field = None

                            effective_date_range_start = (
                                date_range_start if date_range_enabled else None
                            )
                            effective_date_range_end = (
                                date_range_end if date_range_enabled else None
                            )

                            # Validate date range if both bounds are provided
                            if effective_date_range_start and effective_date_range_end:
                                if (
                                    effective_date_range_start
                                    > effective_date_range_end
                                ):
                                    root.after(
                                        0,
                                        lambda t=tgt: log(
                                            f"WARNING: {t} - start_date > end_date, swapping dates"
                                        ),
                                    )
                                    (
                                        effective_date_range_start,
                                        effective_date_range_end,
                                    ) = (
                                        effective_date_range_end,
                                        effective_date_range_start,
                                    )

                            effective_since_date = None if recreate else since_date

                            if recreate:
                                inserted, batches = bulk_insert(
                                    conn,
                                    engine,
                                    tgt,
                                    p,
                                    batch,
                                    schema,
                                    recreate=True,
                                    date_field=effective_date_field,
                                    since_date=effective_since_date,
                                    date_range_start=effective_date_range_start,
                                    date_range_end=effective_date_range_end,
                                )
                            else:
                                inserted, batches = bulk_insert(
                                    conn,
                                    engine,
                                    tgt,
                                    p,
                                    batch,
                                    schema,
                                    recreate=False,
                                    date_field=effective_date_field,
                                    since_date=effective_since_date,
                                    date_range_start=effective_date_range_start,
                                    date_range_end=effective_date_range_end,
                                )
                                if trunc:
                                    truncate_table(conn, engine, tgt, schema)
                                    # When truncating, disable delta sync but keep date range filter if enabled
                                    inserted, batches = bulk_insert(
                                        conn,
                                        engine,
                                        tgt,
                                        p,
                                        batch,
                                        schema,
                                        recreate=False,
                                        date_field=effective_date_field,
                                        since_date=None,
                                        date_range_start=effective_date_range_start,
                                        date_range_end=effective_date_range_end,
                                    )
                            total_rows += inserted
                            root.after(
                                0,
                                lambda b=tgt, i=inserted, bt=batches: log(
                                    f"Loaded {i} rows into {b} in {bt} batch(es)."
                                ),
                            )

                            # Update sync tracking on success
                            if inserted >= 0:
                                # Store ISO with local timezone info
                                sync_updates[table_key] = (
                                    datetime.now().astimezone().isoformat()
                                )
                        except Exception as e:
                            root.after(
                                0,
                                lambda b=base, e=str(e): log(f"Error loading {b}: {e}"),
                            )
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

                # Save updated last sync timestamps
                if sync_updates:
                    sync_tracking.update(sync_updates)
                    save_sync_tracking(sync_tracking)

                root.after(
                    0,
                    lambda: update_progress(
                        100, f"Complete! {total_rows} rows inserted."
                    ),
                )
                root.after(0, lambda: log(f"DONE. Total inserted rows: {total_rows}."))
            except Exception as e:
                root.after(0, lambda: log(f"Upload error: {e}"))
                root.after(0, lambda: update_progress(0, "Error occurred"))
            finally:
                root.after(
                    0,
                    lambda: (
                        upload_btn_ref["btn"].config(state="normal")
                        if upload_btn_ref["btn"]
                        else None
                    ),
                )

        # Start upload in background thread
        _upload_thread = threading.Thread(target=upload_worker, daemon=True)
        _upload_thread.start()

    def run_easy():
        default_data = os.path.join(ksv, "data")
        source_folder = (
            state.get("source_folder") or state.get("folder") or default_data
        )
        if not source_folder or not os.path.isdir(source_folder):
            if os.path.isdir(default_data):
                source_folder = default_data
                folder_var.set(default_data)
            else:
                messagebox.showwarning("Folder", "Select your ksv\\data folder first.")
                return

        # Copy files to sync directory first
        log("Copying DBF files and related files (CDX, FPT) to sync directory...")
        try:
            sync_dir, ffiles = sync_files_to_directory(source_folder)
            log(f"Files copied to sync directory: {sync_dir}")
            state["folder"] = sync_dir
            state["source_folder"] = source_folder

            if not ffiles:
                messagebox.showinfo(
                    "Run (Easy)", "No DBF files found in selected folder."
                )
                return
            files_list.delete(0, "end")
            for f in ffiles:
                files_list.insert("end", os.path.basename(f))
            files_list.selection_clear(0, "end")
            for i in range(len(ffiles)):
                files_list.selection_set(i)
            recreate_var.set(1)
            trunc_var.set(0)
            upload_selected()
        except Exception as e:
            log(f"ERROR: Could not copy files to sync directory: {e}")
            messagebox.showerror(
                "Error", f"Could not copy files to sync directory: {e}"
            )

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
            log("Stopping auto-sync...")
            stop_auto_sync()
            # Wait a moment for thread to stop
            root.after(
                100,
                lambda: (
                    auto_sync_status_var.set("Stopped"),
                    status_label.config(fg=text_color),
                    log("Auto-sync stopped."),
                ),
            )
        else:
            auto_sync_status_var.set("Stopped")
            status_label.config(fg=text_color)
            messagebox.showinfo("Auto-Sync", "Auto-sync is not running.")

    def edit_config_yaml():
        """Open config YAML file in an editor window."""
        cfg_path = default_config_path()

        if not os.path.exists(cfg_path):
            if not messagebox.askyesno(
                "Create Config", "Config file doesn't exist. Create it?"
            ):
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

        tk.Label(
            editor,
            text=f"Editing: {cfg_path}",
            font=label_font,
            bg=bg_color,
            fg=text_color,
        ).pack(pady=5)

        text_frame = tk.Frame(editor, bg=entry_bg, relief="flat", bd=1)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        yaml_text = tk.Text(
            text_frame, wrap="word", font=("Consolas", 9), bg="#fafafa", fg=text_color
        )
        yaml_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=yaml_text.yview)
        yaml_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        # Load current config
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
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
                with open(cfg_path, "w", encoding="utf-8") as f:
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
        ttk.Button(
            btn_frame, text="Save", command=save_yaml, style="Accent.TButton"
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_frame, text="Cancel", command=editor.destroy, style="Modern.TButton"
        ).pack(side="left", padx=5)

    def edit_sync_tracking_yaml():
        """Open sync tracking YAML file in an editor window."""
        tracking_path = default_sync_tracking_path()

        if not os.path.exists(tracking_path):
            # Create empty tracking file if it doesn't exist
            try:
                save_sync_tracking({})
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Could not create sync tracking file: {e}"
                )
                return

        # Create editor window (responsive to screen size)
        editor = tk.Toplevel(root)
        editor.title("Edit Sync Tracking (Last Sync Times)")

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

        tk.Label(
            editor,
            text=f"Editing: {tracking_path}",
            font=label_font,
            bg=bg_color,
            fg=text_color,
        ).pack(pady=5)

        # Add help text
        help_text = tk.Label(
            editor,
            text="Edit last sync timestamps (ISO format: YYYY-MM-DDTHH:MM:SS). Set to a past date to re-sync missed data.",
            font=("Segoe UI", max(7, base_font_size - 2)),
            bg=bg_color,
            fg="#666",
            wraplength=editor_width - 40,
            justify="left",
        )
        help_text.pack(pady=(0, 5))

        text_frame = tk.Frame(editor, bg=entry_bg, relief="flat", bd=1)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        yaml_text = tk.Text(
            text_frame, wrap="word", font=("Consolas", 9), bg="#fafafa", fg=text_color
        )
        yaml_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=yaml_text.yview)
        yaml_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        # Load current sync tracking
        try:
            with open(tracking_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    # Empty file - create default structure
                    content = "# Sync tracking - Last sync timestamps per table\n# Format: YYYY-MM-DDTHH:MM:SS (ISO format)\n# Set to a past date to re-sync missed data\n"
                yaml_text.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load sync tracking: {e}")
            editor.destroy()
            return

        def save_yaml():
            try:
                content = yaml_text.get("1.0", "end-1c")
                # Validate YAML
                data = yaml.safe_load(content)
                if data is None:
                    data = {}
                # Validate date formats (basic check - should be ISO format strings)
                for key, value in data.items():
                    if isinstance(value, str) and key != "__last_auto_sync__":
                        try:
                            # Try to parse as ISO datetime
                            datetime.fromisoformat(value)
                        except ValueError:
                            # Not a valid ISO datetime - warn but allow (might be a comment or other data)
                            pass
                # Save
                with open(tracking_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", "Sync tracking saved successfully!")
                log("Sync tracking YAML saved. Changes will take effect on next sync.")
                editor.destroy()
            except yaml.YAMLError as e:
                messagebox.showerror("YAML Error", f"Invalid YAML:\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save sync tracking: {e}")

        btn_frame = tk.Frame(editor, bg=bg_color)
        btn_frame.pack(pady=10)
        ttk.Button(
            btn_frame, text="Save", command=save_yaml, style="Accent.TButton"
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_frame, text="Cancel", command=editor.destroy, style="Modern.TButton"
        ).pack(side="left", padx=5)

    # --- Modern UI Setup ---
    try:
        root = tk.Tk()
        root.title("VFP DBF → RDS Uploader")
    except Exception as e:
        log_error(f"Error creating Tkinter root window: {e}")
        import traceback

        log_error(traceback.format_exc())
        return

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
    style.theme_use("clam")

    # Style configurations
    style.configure(
        "Title.TLabel", font=title_font, background=bg_color, foreground=header_color
    )
    style.configure(
        "Header.TLabel",
        font=("Segoe UI", 10, "bold"),
        background=bg_color,
        foreground=text_color,
    )
    style.configure("Modern.TButton", font=button_font, padding=8)
    style.configure("Accent.TButton", font=button_font, padding=8)
    style.map(
        "Accent.TButton",
        background=[("active", accent_hover), ("!active", accent_color)],
        foreground=[("active", "white"), ("!active", "white")],
    )

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
    title_label = tk.Label(
        title_frame,
        text="VFP DBF → RDS Uploader",
        font=title_font,
        bg=header_color,
        fg="white",
    )
    title_label.grid(row=0, column=0)
    title_frame.grid_rowconfigure(0, weight=1)

    # Top: folder row with modern styling (compact, fixed height)
    pad_x = max(8, int(window_width / 100))
    pad_y = max(6, int(window_height / 120))
    top = tk.Frame(root, bg=bg_color, padx=pad_x, pady=pad_y)
    top.grid(row=1, column=0, sticky="ew")
    top.grid_columnconfigure(1, weight=1)

    default_folder = (
        os.path.join(ksv, "data") if os.path.isdir(os.path.join(ksv, "data")) else ksv
    )
    folder_var = tk.StringVar(value=default_folder)

    # Responsive folder entry width
    folder_entry_width = max(35, int(window_width / 18))
    tk.Label(top, text="Data Folder", font=label_font, bg=bg_color, fg=text_color).grid(
        row=0, column=0, padx=(0, 6)
    )
    folder_entry = tk.Entry(
        top,
        textvariable=folder_var,
        width=folder_entry_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    folder_entry.grid(row=0, column=1, padx=4, sticky="ew")

    ttk.Button(top, text="Browse", command=browse_folder, style="Modern.TButton").grid(
        row=0, column=2, padx=4
    )
    ttk.Button(top, text="Scan DBFs", command=scan_dbfs, style="Accent.TButton").grid(
        row=0, column=3, padx=4
    )
    count_var = tk.StringVar(value="Found: 0")
    tk.Label(
        top, textvariable=count_var, font=label_font, bg=bg_color, fg=text_color
    ).grid(row=0, column=4, padx=(8, 0))

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

    tk.Label(
        left_frame, text="DBF Tables", font=header_font, bg=bg_color, fg=text_color
    ).grid(row=0, column=0, sticky="w", pady=(0, 3))

    list_container = tk.Frame(left_frame, bg=entry_bg, relief="flat", bd=1)
    list_container.grid(row=1, column=0, sticky="nsew")
    list_container.grid_rowconfigure(0, weight=1)
    list_container.grid_columnconfigure(0, weight=1)

    # Listbox fills available vertical space (no fixed height limit)
    # Calculate reasonable minimum rows based on screen
    min_listbox_rows = max(10, min(int(window_height / 50), 20))
    files_list = tk.Listbox(
        list_container,
        selectmode="extended",
        height=min_listbox_rows,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectbackground=accent_color,
        selectforeground="white",
        relief="flat",
        bd=0,
        highlightthickness=0,
    )
    files_list.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    yscroll = ttk.Scrollbar(list_container, orient="vertical", command=files_list.yview)
    yscroll.grid(row=0, column=1, sticky="ns")
    files_list.configure(yscrollcommand=yscroll.set)

    # Update last auto-sync time display
    def _format_local_time(iso_str: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_str)
            # If timezone-aware, convert to local; else assume local
            if dt.tzinfo is not None:
                dt = dt.astimezone()
            return dt.strftime("%m/%d/%Y %I:%M %p")
        except Exception:
            return iso_str

    def update_last_auto_sync_label():
        """Update the last auto-sync time display from sync tracking."""
        try:
            tracking = load_sync_tracking()
            # Get the most recent auto-sync time from tracking
            # Look for "__last_auto_sync__" key or find most recent sync time
            last_auto_sync = tracking.get("__last_auto_sync__")

            if not last_auto_sync:
                # Fallback: find the most recent sync time from all tables
                sync_times = []
                for key, value in tracking.items():
                    if key != "__last_auto_sync__" and isinstance(value, str):
                        try:
                            dt = datetime.fromisoformat(value)
                            sync_times.append((dt, value))
                        except Exception:
                            continue

                if sync_times:
                    # Get the most recent sync time
                    sync_times.sort(reverse=True)
                    last_auto_sync = sync_times[0][1]

            if last_auto_sync:
                last_sync_var.set(_format_local_time(last_auto_sync))
            else:
                last_sync_var.set("Never")
        except Exception:
            last_sync_var.set("—")

    # Update last sync label periodically
    def periodic_update_last_sync():
        """Periodically update the last auto-sync time display."""
        update_last_auto_sync_label()
        root.after(30000, periodic_update_last_sync)  # Update every 30 seconds

    # Start periodic updates
    root.after(1000, periodic_update_last_sync)  # Initial update after 1 second

    # Right: Form with modern card-like appearance (scrollable for overflow)
    # Create outer frame for scrollable content
    right_outer = tk.Frame(mid, bg=bg_color)
    right_outer.grid(row=0, column=1, sticky="nsew", padx=(pad_x // 2, 0))
    right_outer.grid_rowconfigure(0, weight=1)
    right_outer.grid_columnconfigure(0, weight=1)

    # Create canvas for scrolling
    right_canvas = tk.Canvas(right_outer, bg=entry_bg, highlightthickness=0)
    right_scrollbar = ttk.Scrollbar(
        right_outer, orient="vertical", command=right_canvas.yview
    )
    right_scrollable_frame = tk.Frame(right_canvas, bg=entry_bg)

    # Create the window for the scrollable frame
    canvas_window = right_canvas.create_window(
        (0, 0), window=right_scrollable_frame, anchor="nw"
    )

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
    right_canvas.bind("<Configure>", update_canvas_width)
    right_canvas.configure(yscrollcommand=right_scrollbar.set)

    right_canvas.grid(row=0, column=0, sticky="nsew")
    right_scrollbar.grid(row=0, column=1, sticky="ns")

    # Bind mousewheel to canvas (Windows/Mac)
    def _on_mousewheel(event):
        # Only scroll if mouse is over the canvas
        if right_canvas.winfo_containing(event.x_root, event.y_root):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

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
    right_frame.configure(
        relief="flat", bd=1, padx=max(8, pad_x), pady=max(6, pad_y // 2)
    )
    right_frame.grid_columnconfigure(1, weight=1)

    # Helper function to create collapsible sections
    def create_collapsible_section(
        parent, title, start_row, bg_color=entry_bg, fg_color=header_color
    ):
        """Create a collapsible section with expand/collapse functionality."""
        # Container frame for the section
        section_frame = tk.Frame(parent, bg=bg_color)
        section_frame.grid(
            row=start_row, column=0, columnspan=2, sticky="ew", pady=(0, 3)
        )
        section_frame.grid_columnconfigure(1, weight=1)

        # Header with clickable label
        header_frame = tk.Frame(section_frame, bg=bg_color, cursor="hand2")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        # Expand/collapse indicator
        expand_var = tk.BooleanVar(value=False)  # Start collapsed
        expand_label = tk.Label(
            header_frame,
            text="▶",
            font=("Segoe UI", 8),
            bg=bg_color,
            fg=fg_color,
            cursor="hand2",
        )
        expand_label.grid(row=0, column=0, padx=(0, 4))

        # Title label
        title_label = tk.Label(
            header_frame,
            text=title,
            font=header_font,
            bg=bg_color,
            fg=fg_color,
            cursor="hand2",
        )
        title_label.grid(row=0, column=1, sticky="w")

        # Content frame (initially hidden - collapsed)
        content_frame = tk.Frame(section_frame, bg=bg_color)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_remove()  # Start collapsed

        def toggle_section():
            if expand_var.get():
                # Collapse
                content_frame.grid_remove()
                expand_label.config(text="▶")
                expand_var.set(False)
            else:
                # Expand
                content_frame.grid()
                expand_label.config(text="▼")
                expand_var.set(True)
            update_scroll_region()

        # Bind click events to header
        for widget in [header_frame, expand_label, title_label]:
            widget.bind("<Button-1>", lambda e: toggle_section())

        return content_frame, start_row + 1

    row = 0
    # Connection Settings (collapsible)
    conn_content_frame, next_row = create_collapsible_section(
        right_frame, "Connection Settings", row
    )
    row = next_row

    engine_var = tk.StringVar(value="mysql")
    tk.Label(
        conn_content_frame, text="Engine", font=label_font, bg=entry_bg, fg=text_color
    ).grid(row=0, column=0, sticky="w", pady=4)
    engine_menu = ttk.OptionMenu(
        conn_content_frame, engine_var, "mysql", "mysql", "mssql"
    )
    engine_menu.grid(row=0, column=1, sticky="we", padx=4, pady=4)

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

    # Date range filter variables
    date_range_enabled_var = tk.IntVar(value=0)
    date_range_field_var = tk.StringVar()
    date_range_start_var = tk.StringVar()
    date_range_end_var = tk.StringVar()

    # Admin API variables
    admin_enabled_var = tk.IntVar(value=0)
    admin_base_url_var = tk.StringVar()
    admin_api_key_var = tk.StringVar()
    admin_store_id_var = tk.StringVar()

    # Responsive entry field width
    # Entry width based on window width (settings panel is ~40% of window)
    entry_width = max(20, int(window_width * 0.35 / 12))
    # Date field width (same as start/end date entries) - make wider to prevent text cutoff
    # Increased to ensure "Tstamp", "updated", "created", "cdate" etc. are fully visible
    # Minimum 25 characters to accommodate longer field names
    date_field_width = max(25, int(entry_width * 1.1))

    conn_row = 1
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
        tk.Label(
            conn_content_frame, text=label, font=label_font, bg=entry_bg, fg=text_color
        ).grid(row=conn_row, column=0, sticky="w", pady=row_pad)
        entry = tk.Entry(
            conn_content_frame,
            textvariable=var,
            width=width,
            font=label_font,
            bg=entry_bg,
            relief="flat",
            bd=1,
            highlightthickness=1,
            highlightbackground="#ddd",
            highlightcolor=accent_color,
        )
        if is_password:
            entry.config(show="*")
        entry.grid(row=conn_row, column=1, sticky="we", padx=3, pady=row_pad)
        manual_db_widgets.append(entry)
        conn_row += 1

    tk.Label(
        conn_content_frame,
        text="Schema (mssql)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=conn_row, column=0, sticky="w", pady=row_pad)
    schema_entry = tk.Entry(
        conn_content_frame,
        textvariable=schema_var,
        width=entry_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    schema_entry.grid(row=conn_row, column=1, sticky="we", padx=3, pady=row_pad)
    manual_db_widgets.append(engine_menu)
    manual_db_widgets.append(schema_entry)
    conn_row += 1

    tk.Checkbutton(
        conn_content_frame,
        text="Drop & recreate tables (safe)",
        variable=recreate_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectcolor=entry_bg,
        activebackground=entry_bg,
        activeforeground=text_color,
    ).grid(row=conn_row, column=0, columnspan=2, sticky="w", pady=row_pad)
    conn_row += 1
    tk.Checkbutton(
        conn_content_frame,
        text="Truncate before load",
        variable=trunc_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectcolor=entry_bg,
        activebackground=entry_bg,
        activeforeground=text_color,
    ).grid(row=conn_row, column=0, columnspan=2, sticky="w", pady=row_pad)
    conn_row += 1

    tk.Label(
        conn_content_frame,
        text="Batch size",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=conn_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        conn_content_frame,
        textvariable=batch_var,
        width=max(8, entry_width // 2),
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=conn_row, column=1, sticky="w", padx=3, pady=row_pad)
    conn_row += 1

    tk.Label(
        conn_content_frame,
        text="Table prefix",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=conn_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        conn_content_frame,
        textvariable=tpref_var,
        width=max(15, int(entry_width * 0.7)),
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=conn_row, column=1, sticky="w", padx=3, pady=row_pad)
    conn_row += 1

    tk.Label(
        conn_content_frame,
        text="Table suffix",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=conn_row, column=0, sticky="w", pady=row_pad)
    tsuff_entry = tk.Entry(
        conn_content_frame,
        textvariable=tsuff_var,
        width=max(15, int(entry_width * 0.7)),
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    tsuff_entry.grid(row=conn_row, column=1, sticky="w", padx=3, pady=row_pad)

    # Admin Backend (collapsible)
    admin_content_frame, next_row = create_collapsible_section(
        right_frame, "Admin Backend", row
    )
    row = next_row

    admin_row = 0
    tk.Checkbutton(
        admin_content_frame,
        text="Use Admin Backend creds",
        variable=admin_enabled_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectcolor=entry_bg,
        activebackground=entry_bg,
        activeforeground=text_color,
    ).grid(row=admin_row, column=0, columnspan=2, sticky="w", pady=row_pad)
    admin_row += 1

    tk.Label(
        admin_content_frame,
        text="Base URL",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=admin_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        admin_content_frame,
        textvariable=admin_base_url_var,
        width=entry_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=admin_row, column=1, sticky="we", padx=3, pady=row_pad)
    admin_row += 1

    tk.Label(
        admin_content_frame, text="API key", font=label_font, bg=entry_bg, fg=text_color
    ).grid(row=admin_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        admin_content_frame,
        textvariable=admin_api_key_var,
        width=entry_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=admin_row, column=1, sticky="we", padx=3, pady=row_pad)
    admin_row += 1

    tk.Label(
        admin_content_frame,
        text="Store ID",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=admin_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        admin_content_frame,
        textvariable=admin_store_id_var,
        width=max(12, entry_width // 2),
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=admin_row, column=1, sticky="w", padx=3, pady=row_pad)

    # Date Range Filter section (collapsible)
    date_range_content_frame, next_row = create_collapsible_section(
        right_frame, "Date Range Filter", row
    )
    row = next_row

    date_range_row = 0
    tk.Checkbutton(
        date_range_content_frame,
        text="Enable date range filter (for initial loads)",
        variable=date_range_enabled_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectcolor=entry_bg,
        activebackground=entry_bg,
        activeforeground=text_color,
    ).grid(row=date_range_row, column=0, columnspan=2, sticky="w", pady=row_pad)
    date_range_row += 1
    # Add explanation text (on separate row below checkbox)
    explain_label = tk.Label(
        date_range_content_frame,
        text="Filters rows where the date field column is between start and end dates. Tables without the date field are uploaded entirely.",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    explain_label.grid(
        row=date_range_row,
        column=0,
        columnspan=2,
        sticky="w",
        padx=(20, 0),
        pady=(0, row_pad),
    )
    date_range_row += 1

    tk.Label(
        date_range_content_frame,
        text="Date field (column name in DBF)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=date_range_row, column=0, sticky="w", pady=row_pad)
    date_range_field_entry = tk.Entry(
        date_range_content_frame,
        textvariable=date_range_field_var,
        width=date_field_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    date_range_field_entry.grid(
        row=date_range_row, column=1, sticky="we", padx=3, pady=row_pad
    )
    date_range_row += 1
    # Add tooltip-style help text (on next row to not interfere with entry box)
    help_label = tk.Label(
        date_range_content_frame,
        text="(e.g., 'date', 'saledate', 'created' - name of the date column in your DBF table)",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    help_label.grid(
        row=date_range_row,
        column=0,
        columnspan=2,
        sticky="w",
        padx=(20, 0),
        pady=(0, row_pad),
    )
    date_range_row += 1

    tk.Label(
        date_range_content_frame,
        text="Start date (YYYY-MM-DD)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=date_range_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        date_range_content_frame,
        textvariable=date_range_start_var,
        width=date_field_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=date_range_row, column=1, sticky="we", padx=3, pady=row_pad)
    date_range_row += 1

    tk.Label(
        date_range_content_frame,
        text="End date (YYYY-MM-DD)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=date_range_row, column=0, sticky="w", pady=row_pad)
    tk.Entry(
        date_range_content_frame,
        textvariable=date_range_end_var,
        width=date_field_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    ).grid(row=date_range_row, column=1, sticky="we", padx=3, pady=row_pad)

    # Delta sync section (collapsible)
    delta_content_frame, next_row = create_collapsible_section(
        right_frame, "Delta Sync", row
    )
    row = next_row

    delta_row = 0
    tk.Checkbutton(
        delta_content_frame,
        text="Enable delta sync",
        variable=delta_enabled_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
        selectcolor=entry_bg,
        activebackground=entry_bg,
        activeforeground=text_color,
    ).grid(row=delta_row, column=0, columnspan=2, sticky="w", pady=row_pad)
    delta_row += 1
    # Add explanation for delta sync (on separate row below checkbox)
    delta_explain_label = tk.Label(
        delta_content_frame,
        text="Only syncs records where the date field is newer than the last sync time (based on system time). Tracks last sync per table automatically. Use with Auto-Sync below to run on a schedule (hourly, etc.).",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    delta_explain_label.grid(
        row=delta_row,
        column=0,
        columnspan=2,
        sticky="w",
        padx=(20, 0),
        pady=(0, row_pad),
    )
    delta_row += 1

    tk.Label(
        delta_content_frame,
        text="Date field (column name in DBF)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=delta_row, column=0, sticky="w", pady=row_pad)
    date_entry = tk.Entry(
        delta_content_frame,
        textvariable=delta_date_field_var,
        width=date_field_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    date_entry.grid(row=delta_row, column=1, sticky="we", padx=3, pady=row_pad)
    delta_row += 1
    # Add tooltip-style help text for delta sync date field (on next row to not interfere with entry box)
    delta_help_label = tk.Label(
        delta_content_frame,
        text="(e.g., 'tstamp', 'time', 'created' - used to find records newer than last sync)",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    delta_help_label.grid(
        row=delta_row,
        column=0,
        columnspan=2,
        sticky="w",
        padx=(20, 0),
        pady=(0, row_pad),
    )
    delta_row += 1

    tk.Label(
        delta_content_frame,
        text="Auto-sync interval (seconds)",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=delta_row, column=0, sticky="w", pady=row_pad)
    auto_interval_entry = tk.Entry(
        delta_content_frame,
        textvariable=delta_interval_var,
        width=date_field_width,
        font=label_font,
        bg=entry_bg,
        relief="flat",
        bd=1,
        highlightthickness=1,
        highlightbackground="#ddd",
        highlightcolor=accent_color,
    )
    auto_interval_entry.grid(row=delta_row, column=1, sticky="we", padx=3, pady=row_pad)
    delta_row += 1
    # Add help text for auto-sync interval (on next row to not interfere with entry box)
    auto_interval_help = tk.Label(
        delta_content_frame,
        text="(e.g., 3600 = hourly, 1800 = every 30 min, 86400 = daily. Set to 0 to disable auto-sync)",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    auto_interval_help.grid(
        row=delta_row,
        column=0,
        columnspan=2,
        sticky="w",
        padx=(20, 0),
        pady=(0, row_pad),
    )

    # Show last sync time for selected table (collapsible)
    last_sync_frame, next_row = create_collapsible_section(
        right_frame, "Last Sync", row, bg_color=entry_bg
    )
    row = next_row
    tk.Label(
        last_sync_frame,
        text="Last auto-sync",
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=0, column=0, sticky="w", pady=row_pad)
    tk.Label(
        last_sync_frame,
        textvariable=last_sync_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    ).grid(row=0, column=1, sticky="w", padx=3, pady=row_pad)

    # Auto-sync status and controls (collapsible)
    auto_sync_content_frame, next_row = create_collapsible_section(
        right_frame, "Auto-Sync (Time-Based)", row
    )
    row = next_row

    auto_sync_row = 0
    auto_sync_explain = tk.Label(
        auto_sync_content_frame,
        text="Automatically runs delta sync on a schedule based on the interval above. Uses system time to track updates.",
        font=("Segoe UI", max(7, base_font_size - 2)),
        bg=entry_bg,
        fg="#666",
        wraplength=400,
        justify="left",
    )
    auto_sync_explain.grid(
        row=auto_sync_row, column=0, columnspan=2, sticky="w", padx=(0, 0), pady=(0, 4)
    )
    auto_sync_row += 1

    auto_sync_frame = tk.Frame(auto_sync_content_frame, bg=entry_bg)
    auto_sync_frame.grid(
        row=auto_sync_row, column=0, columnspan=2, sticky="we", pady=(4, 0)
    )

    tk.Label(
        auto_sync_frame, text="Status:", font=label_font, bg=entry_bg, fg=text_color
    ).pack(side="left", padx=(0, 5))
    status_label = tk.Label(
        auto_sync_frame,
        textvariable=auto_sync_status_var,
        font=label_font,
        bg=entry_bg,
        fg=text_color,
    )
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

    ttk.Button(
        auto_sync_frame,
        text="Start",
        command=start_auto_sync_gui,
        style="Modern.TButton",
        width=8,
    ).pack(side="left", padx=2)
    ttk.Button(
        auto_sync_frame,
        text="Stop",
        command=stop_auto_sync_gui,
        style="Modern.TButton",
        width=8,
    ).pack(side="left", padx=2)
    row += 1

    # Button frame with modern styling (reduced padding)
    btn_pad = 4
    btn_frame = tk.Frame(right_frame, bg=entry_bg)
    btn_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(btn_pad, 0))

    ttk.Button(
        btn_frame, text="Load Config", command=load_cfg, style="Modern.TButton"
    ).pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(
        btn_frame, text="Save Config", command=save_cfg, style="Modern.TButton"
    ).pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(
        btn_frame, text="Edit YAML", command=edit_config_yaml, style="Modern.TButton"
    ).pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(
        btn_frame,
        text="Edit Sync Times",
        command=edit_sync_tracking_yaml,
        style="Modern.TButton",
    ).pack(side="left", padx=2, fill="x", expand=True)

    action_frame = tk.Frame(right_frame, bg=entry_bg)
    action_frame.grid(
        row=row + 1, column=0, columnspan=2, sticky="we", pady=(btn_pad, 0)
    )

    # Create upload button now that upload_selected and action_frame are both defined
    upload_btn_ref["btn"] = ttk.Button(
        action_frame,
        text="Upload Selected",
        command=upload_selected,
        style="Accent.TButton",
    )
    upload_btn_ref["btn"].pack(side="left", padx=2, fill="x", expand=True)
    ttk.Button(
        action_frame, text="Run (Easy)", command=run_easy, style="Accent.TButton"
    ).pack(side="left", padx=2, fill="x", expand=True)

    right_frame.grid_columnconfigure(1, weight=1)

    # Disable/enable manual DB fields based on admin toggle
    def _apply_admin_toggle(*_):
        use_admin = bool(admin_enabled_var.get())
        state = "disabled" if use_admin else "normal"
        # Engine menu is a ttk OptionMenu which doesn't support state directly; skip it
        for w in manual_db_widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass

    admin_enabled_var.trace_add("write", _apply_admin_toggle)
    # Apply initial
    _apply_admin_toggle()

    # Log box with modern styling (guaranteed minimum height, always visible)
    log_frame = tk.Frame(root, bg=bg_color, padx=pad_x, pady=max(4, pad_y // 2))
    log_frame.grid(row=3, column=0, sticky="ew")
    log_frame.grid_columnconfigure(0, weight=1)
    log_frame.grid_rowconfigure(1, weight=1)  # Log container can grow
    log_frame.grid_rowconfigure(2, weight=0)  # Progress bar row (fixed height)

    # Calculate log height as percentage of window (ensures visibility)
    # Minimum 15% of window height, maximum 25%
    log_min_px = max(80, int(window_height * 0.15))  # Always at least 15% of window
    log_max_px = min(int(window_height * 0.25), 200)  # Max 25% or 200px
    log_height_px = max(log_min_px, min(log_max_px, int(window_height * 0.2)))

    tk.Label(
        log_frame, text="Activity Log", font=header_font, bg=bg_color, fg=text_color
    ).grid(row=0, column=0, sticky="w", pady=(0, 3))

    log_container = tk.Frame(log_frame, bg=entry_bg, relief="flat", bd=1)
    log_container.grid(row=1, column=0, sticky="ew")
    log_container.config(height=log_height_px)
    log_container.grid_propagate(False)  # Maintain minimum height
    log_container.grid_rowconfigure(0, weight=1)
    log_container.grid_columnconfigure(0, weight=1)

    log_font_size = max(8, base_font_size - 1)
    # Calculate rows based on pixel height (approx 20px per row)
    log_rows = max(8, int(log_height_px / 22))
    log_box = tk.Text(
        log_container,
        height=log_rows,
        wrap="word",
        font=("Consolas", log_font_size),
        bg="#fafafa",
        fg=text_color,
        relief="flat",
        bd=0,
        padx=max(6, pad_x // 2),
        pady=max(4, pad_y // 2),
    )
    log_box.grid(row=0, column=0, sticky="nsew")
    log_box.configure(state="disabled")

    log_scroll = ttk.Scrollbar(log_container, orient="vertical", command=log_box.yview)
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_box.configure(yscrollcommand=log_scroll.set)

    # Progress bar variables
    progress_var = tk.DoubleVar(value=0.0)
    progress_status_var = tk.StringVar(value="Ready")
    progress_percent_var = tk.StringVar(value="0%")

    # Progress bar widget (initially hidden)
    progress_frame = tk.Frame(log_frame, bg=bg_color)
    progress_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    progress_frame.grid_columnconfigure(0, weight=1)

    progress_label = tk.Label(
        progress_frame,
        textvariable=progress_status_var,
        font=label_font,
        bg=bg_color,
        fg=text_color,
    )
    progress_label.grid(row=0, column=0, sticky="w", pady=(0, 2))

    # Progress bar with percentage label
    progress_bar_container = tk.Frame(progress_frame, bg=bg_color)
    progress_bar_container.grid(row=1, column=0, sticky="ew", pady=(0, 2))
    progress_frame.grid_columnconfigure(0, weight=1)
    progress_bar_container.grid_columnconfigure(0, weight=1)

    progress_bar = ttk.Progressbar(
        progress_bar_container,
        variable=progress_var,
        maximum=100,
        length=300,
        mode="determinate",
    )
    progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 8))

    # Percentage label next to progress bar
    progress_percent_label = tk.Label(
        progress_bar_container,
        textvariable=progress_percent_var,
        font=label_font,
        bg=bg_color,
        fg=text_color,
        width=5,
    )
    progress_percent_label.grid(row=0, column=1, sticky="e")

    # Hide progress bar initially
    progress_frame.grid_remove()

    # Auto-load config on launch if it exists
    if os.path.exists(default_config_path()):
        try:
            load_cfg()
            log("Config loaded automatically on launch.")
        except Exception as e:
            log(f"Could not auto-load config: {e}")

    # Auto-scan default folder on open
    scan_dbfs()
    # Initialize last auto-sync label (will be updated by periodic_update_last_sync)

    # Start status update timer
    update_auto_sync_status()

    # Set global log and progress functions after all widgets are created
    _gui_log_func = log
    _gui_progress_func = update_progress
    _gui_root = root

    # System tray setup
    tray_icon = None
    tray_thread = None

    def setup_system_tray():
        """Setup system tray icon for minimize to tray functionality."""
        try:
            import pystray
            from PIL import Image, ImageDraw
            import threading

            # Create a simple icon
            image = Image.new("RGB", (64, 64), color="white")
            draw = ImageDraw.Draw(image)
            draw.ellipse([16, 16, 48, 48], fill="blue")
            draw.text((20, 26), "VFP", fill="white")

            def show_window(icon=None, item=None):
                """Show and restore the window."""
                root.after(0, lambda: root.deiconify())
                root.after(0, lambda: root.lift())
                root.after(0, lambda: root.focus_force())

            def quit_app(icon=None, item=None):
                """Quit the application."""
                root.after(0, lambda: root.quit())
                if tray_icon:
                    tray_icon.stop()

            menu = pystray.Menu(
                pystray.MenuItem("Show", show_window),
                pystray.MenuItem("Quit", quit_app),
            )

            icon = pystray.Icon(
                "VFP_DBF_Uploader", image, "VFP DBF → RDS Uploader", menu
            )
            icon.on_click = show_window  # Click to restore

            return icon
        except ImportError:
            # pystray not available, return None
            return None

    def on_minimize(event=None):
        """Handle window minimize - hide to system tray."""
        if tray_icon:
            root.withdraw()  # Hide window to system tray
            log("Window minimized to system tray. Click tray icon to restore.")

    def on_close(event=None):
        """Handle window close - actually quit the program."""
        # Clean up and close
        cleanup_instance_lock(lock_handle)
        if tray_icon:
            try:
                tray_icon.stop()
            except Exception:
                pass
        root.quit()
        return None

    # Setup system tray if available (but don't hide window immediately)
    try:
        tray_icon = setup_system_tray()
        if tray_icon:
            # Run tray icon in separate thread
            def run_tray():
                try:
                    tray_icon.run()
                except Exception:
                    pass

            tray_thread = threading.Thread(target=run_tray, daemon=True)
            tray_thread.start()

            # Bind close event (X button - actually closes)
            root.protocol("WM_DELETE_WINDOW", on_close)

            # Bind minimize event to hide to system tray
            # BUT: Only hide if user explicitly minimizes (not on startup)
            def on_minimize_event(event=None):
                """Handle minimize button - hide to system tray."""
                # Only hide if window is actually minimized by user action
                # Don't hide on startup
                if tray_icon and root.state() == "iconic":
                    # Small delay to ensure it's a real minimize action
                    root.after(
                        100,
                        lambda: root.withdraw() if root.state() == "iconic" else None,
                    )
                    log("Window minimized to system tray. Click tray icon to restore.")

            # Monitor for minimize events - but only AFTER window is shown for a few seconds
            # This prevents immediately hiding the window on startup
            def check_minimize():
                """Periodically check if window was minimized."""
                # Only check if window has been shown for at least 3 seconds
                # This prevents immediately hiding the window on startup
                if not hasattr(check_minimize, "start_time"):
                    check_minimize.start_time = time.time()

                elapsed = time.time() - check_minimize.start_time
                if elapsed < 3.0:
                    # Window just started - don't hide it yet, wait a bit longer
                    root.after(1000, check_minimize)
                    return

                # After 3 seconds, it's safe to check for minimize
                if tray_icon and root.state() == "iconic" and root.winfo_viewable():
                    on_minimize_event()
                root.after(1000, check_minimize)

            # Delay the minimize check significantly - give window plenty of time to show first
            root.after(3000, check_minimize)  # Start checking after 3 seconds
    except Exception as e:
        log_error(f"System tray setup error: {e}")

        # System tray not available, just handle normal close
        def on_close_normal(event=None):
            cleanup_instance_lock(lock_handle)
            root.quit()

        root.protocol("WM_DELETE_WINDOW", on_close_normal)

    try:
        # Force window to be visible and on top before starting mainloop
        root.update_idletasks()  # Process all pending events
        root.update()  # Update display
        root.deiconify()  # Ensure window is not minimized
        root.lift()  # Bring to front
        root.attributes("-topmost", True)  # Force to top
        root.update()
        root.attributes("-topmost", False)  # Remove topmost after showing
        root.focus_force()  # Force focus

        # Ensure window is actually visible
        if not root.winfo_viewable():
            root.deiconify()

        root.mainloop()
    except Exception as e:
        log_error(f"Error in mainloop: {e}")
        import traceback

        log_error(traceback.format_exc())
    finally:
        # Cleanup on exit
        try:
            cleanup_instance_lock(lock_handle)
        except Exception:
            pass
        if tray_icon:
            try:
                tray_icon.stop()
            except Exception:
                pass


# ---------- Auto-sync functionality ----------

_auto_sync_thread = None
_auto_sync_stop = threading.Event()


def run_auto_sync(cfg_path: Optional[str] = None, profile: Optional[str] = None):
    """
    Unified auto-sync and delta sync function.

    Runs on a schedule and automatically performs delta sync:
    - Checks timestamp field in each table
    - Only uploads records with timestamps newer than last sync time
    - Updates sync timestamp after each successful sync
    - Runs continuously on configured interval
    """
    configure_logging()

    raw_cfg = load_config(cfg_path) if cfg_path else load_config()
    cfg = resolve_profile(raw_cfg, profile)

    delta_cfg = cfg.get("delta_sync", {})
    interval_seconds = int(
        delta_cfg.get("auto_sync_interval_seconds", 3600)
    )  # Default 1 hour
    delta_enabled = bool(delta_cfg.get("enabled", False))
    date_field = delta_cfg.get(
        "date_field"
    )  # Timestamp field to check (e.g., 'tstamp', 'date', 'updated')

    if not delta_enabled:
        log_to_gui(
            "WARNING: Auto-sync requires delta_sync.enabled=true. Auto-sync will do full syncs."
        )
    elif not date_field:
        log_to_gui(
            "WARNING: Auto-sync requires delta_sync.date_field to be set. Auto-sync will do full syncs."
        )

    log_to_gui(
        f"Starting auto-sync (interval: {interval_seconds}s, profile: {profile or 'default'})"
    )
    if delta_enabled and date_field:
        log_to_gui(
            f"Delta sync enabled: checking '{date_field}' field for new records since last sync"
        )
    log_to_gui("Auto-sync started. Press Stop button or close application to stop.")
    update_gui_progress(0, "Auto-sync running...")

    while not _auto_sync_stop.is_set():
        try:
            sync_start_time = datetime.now()
            log_to_gui(
                f"\n[{sync_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Running auto-sync..."
            )
            update_gui_progress(5, "Starting sync cycle...")

            # Run sync with delta sync automatically enabled if configured
            # When auto_sync=True, run_headless will automatically use delta sync
            run_headless(cfg_path, profile, auto_sync=True)

            sync_end_time = datetime.now()
            duration = (sync_end_time - sync_start_time).total_seconds()

            # Update last auto-sync time in tracking
            sync_tracking = load_sync_tracking()
            sync_tracking["__last_auto_sync__"] = sync_end_time.astimezone().isoformat()
            save_sync_tracking(sync_tracking)

            log_to_gui(
                f"[{sync_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Sync complete (took {duration:.1f}s). Next sync in {interval_seconds}s"
            )
            update_gui_progress(
                100, f"Sync complete. Next sync in {interval_seconds}s..."
            )
        except KeyboardInterrupt:
            log_to_gui("\nStopping auto-sync...")
            break
        except Exception as e:
            log_to_gui(f"Error during auto-sync: {e}")
            import traceback

            traceback.print_exc()
            log_to_gui(f"Retrying in {interval_seconds}s...")
            update_gui_progress(
                0, f"Error occurred. Retrying in {interval_seconds}s..."
            )

        # Wait for interval, but check stop event periodically
        for i in range(interval_seconds):
            if _auto_sync_stop.wait(timeout=1):
                log_to_gui("Auto-sync stop requested by user.")
                break
            # Update progress during wait (show countdown)
            if i % 60 == 0:  # Update every minute
                remaining = interval_seconds - i
                update_gui_progress(0, f"Waiting... Next sync in {remaining}s...")

    log_to_gui("Auto-sync stopped.")
    update_gui_progress(0, "Auto-sync stopped.")


def start_auto_sync_background(
    cfg_path: Optional[str] = None, profile: Optional[str] = None
):
    """Start auto-sync in a background thread."""
    global _auto_sync_thread
    if _auto_sync_thread and _auto_sync_thread.is_alive():
        print("Auto-sync already running.")
        return

    _auto_sync_stop.clear()
    _auto_sync_thread = threading.Thread(
        target=run_auto_sync, args=(cfg_path, profile), daemon=True
    )
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

    configure_logging()

    ap = argparse.ArgumentParser(description="VFP DBF → RDS Uploader")
    ap.add_argument(
        "--config",
        help="Path to YAML config (defaults to ksv\\vfp_uploader.yaml if found, else AppData)",
    )
    ap.add_argument(
        "--init",
        action="store_true",
        help="Run interactive setup wizard and save config",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Launch Tkinter GUI (default if no other mode specified)",
    )
    ap.add_argument("--dpg", action="store_true", help="Launch DearPyGui GUI")
    ap.add_argument("--profile", help="Profile name in config (when using profiles)")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for Task Scheduler). Use with --auto-sync for scheduled runs.",
    )
    ap.add_argument(
        "--silent",
        action="store_true",
        help="Run silently without console output (use with --headless)",
    )
    ap.add_argument(
        "--auto-sync",
        action="store_true",
        help="Run auto-sync (periodic sync based on config interval). Implies --headless.",
    )
    ap.add_argument("--stop-sync", action="store_true", help="Stop running auto-sync")
    args = ap.parse_args()

    # Redirect output if silent mode
    if args.silent and args.headless:
        import sys
        import os

        # Redirect stdout and stderr to null device
        try:
            devnull = open(os.devnull, "w")
            sys.stdout = devnull
            sys.stderr = devnull
        except Exception:
            pass  # If we can't redirect, continue anyway

    if args.stop_sync:
        stop_auto_sync()
        return

    if args.init:
        cli_init(args.config)
        run_headless(args.config, profile=args.profile)
        return

    # Auto-sync implies headless mode (for Task Scheduler)
    if args.auto_sync:
        try:
            run_auto_sync(args.config, profile=args.profile)
        except KeyboardInterrupt:
            if not args.silent:
                print("\nStopped by user.")
        return

    # Explicit GUI modes
    if args.dpg:
        run_gui()
        return

    if args.gui:
        run_gui_tk()
        return

    # Headless mode (explicit)
    if args.headless:
        cfg_path = args.config or default_config_path()
        if not Path(cfg_path).exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
        debug_config_path("Resolved config", cfg_path)
        run_headless(cfg_path, profile=args.profile)
        return

    # Default behavior: Launch GUI (when double-clicked or run without arguments)
    run_gui_tk()
    return


if __name__ == "__main__":
    main()
