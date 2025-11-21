from datetime import datetime
from typing import Any, Dict, List, Optional
import sys
import os
import logging

import mysql.connector
from mysql.connector import errorcode
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks

from app.auth import get_auth_user
from app.db_core import get_core_connection

router = APIRouter()

# Add scripts directory to path to import vfp_dbf_to_rdsv2
# Get absolute path to scripts directory
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
scripts_dir = os.path.join(project_root, "scripts")
scripts_dir = os.path.abspath(scripts_dir)

# Verify the script file exists
vfp_script_path = os.path.join(scripts_dir, "vfp_dbf_to_rdsv2.py")

# Try alternative paths if primary doesn't exist
if not os.path.exists(vfp_script_path):
    # Try relative to current working directory
    alt_scripts_dir = os.path.abspath("scripts")
    alt_vfp_script_path = os.path.join(alt_scripts_dir, "vfp_dbf_to_rdsv2.py")
    if os.path.exists(alt_vfp_script_path):
        scripts_dir = alt_scripts_dir
        vfp_script_path = alt_vfp_script_path
        logging.info(f"Using alternative scripts path: {scripts_dir}")
    else:
        # Try environment variable if set
        env_scripts_dir = os.getenv("VFP_SCRIPTS_DIR")
        if env_scripts_dir and os.path.exists(os.path.join(env_scripts_dir, "vfp_dbf_to_rdsv2.py")):
            scripts_dir = os.path.abspath(env_scripts_dir)
            vfp_script_path = os.path.join(scripts_dir, "vfp_dbf_to_rdsv2.py")
            logging.info(f"Using scripts path from environment: {scripts_dir}")
        else:
            logging.warning(f"VFP script not found at: {vfp_script_path}")
            logging.warning(f"Also tried: {alt_vfp_script_path}")
            if env_scripts_dir:
                logging.warning(f"Also tried environment path: {env_scripts_dir}")

# Try to import if script file exists
run_headless = None
if os.path.exists(vfp_script_path):
    # Add to path if not already there
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    try:
        from vfp_dbf_to_rdsv2 import run_headless
        logging.info(f"Successfully imported vfp_dbf_to_rdsv2 from {scripts_dir}")
    except ImportError as e:
        logging.error(f"Could not import vfp_dbf_to_rdsv2 from {scripts_dir}: {e}", exc_info=True)
        run_headless = None
    except Exception as e:
        logging.error(f"Unexpected error importing vfp_dbf_to_rdsv2: {e}", exc_info=True)
        run_headless = None
else:
    logging.error(f"VFP script file does not exist at any checked location")


@router.get("/status")
def get_sync_status(
    store: Optional[str] = Query(default=None, description="Filter sync logs by store identifier"),
    user: dict = Depends(get_auth_user),
) -> Dict[str, Any]:
    stores = user.get("stores") or []
    if not stores and user.get("store_db"):
        stores = [
            {
                "store_db": user.get("store_db"),
                "db_user": user.get("db_user"),
                "db_pass": user.get("db_pass"),
                "store_id": user.get("store_id"),
                "store_name": user.get("store_name"),
            }
        ]

    if not stores:
        raise HTTPException(status_code=403, detail="No stores associated with this account")

    if store:
        matching = [s for s in stores if store in {str(s.get("store_id")), s.get("store_db")}]
        if not matching:
            raise HTTPException(status_code=404, detail="Requested store is not assigned to this user")
        stores = matching

    store_dbs = [s.get("store_db") for s in stores if s.get("store_db")]
    store_lookup = {s.get("store_db"): s.get("store_name") for s in stores if s.get("store_db")}
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        params: List[Any] = []
        where_clause = ""
        if store_dbs:
            placeholders = ",".join(["%s"] * len(store_dbs))
            where_clause = f"WHERE store_db IN ({placeholders})"
            params.extend(store_dbs)

        cursor.execute(
            f"""
            SELECT id, store_db, status, started_at, finished_at, records_processed, message
            FROM dbf_sync_logs
            {where_clause}
            ORDER BY started_at DESC
            LIMIT 100
            """,
            tuple(params),
        )
        rows = cursor.fetchall() or []

        def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
            serialized = dict(row)
            store_name = store_lookup.get(serialized.get("store_db"))
            if store_name:
                serialized.setdefault("store_name", store_name)
            for key in ("started_at", "finished_at"):
                value = serialized.get(key)
                if isinstance(value, datetime):
                    serialized[key] = value.isoformat()
            if serialized.get("records_processed") is not None:
                try:
                    serialized["records_processed"] = int(serialized["records_processed"])
                except (TypeError, ValueError):
                    pass
            return serialized

        return {"logs": [_serialize(row) for row in rows]}
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_NO_SUCH_TABLE:
            return {"logs": []}
        raise HTTPException(status_code=500, detail=f"Failed to load sync status: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sync status: {exc}")
    finally:
        cursor.close()
        conn.close()


def _get_store_info(user: dict, store_identifier: Optional[str] = None) -> Dict[str, Any]:
    """Extract store information from user token, optionally filtered by store identifier."""
    stores = user.get("stores") or []
    if not stores and user.get("store_db"):
        stores = [
            {
                "store_db": user.get("store_db"),
                "db_user": user.get("db_user"),
                "db_pass": user.get("db_pass"),
                "store_id": user.get("store_id"),
                "store_name": user.get("store_name"),
            }
        ]

    if not stores:
        raise HTTPException(status_code=403, detail="No stores associated with this account")

    if store_identifier:
        matching = [
            s
            for s in stores
            if store_identifier in {str(s.get("store_id")), s.get("store_db")}
        ]
        if not matching:
            raise HTTPException(
                status_code=404, detail="Requested store is not assigned to this user"
            )
        return matching[0]

    # Return first store if no identifier provided
    return stores[0]


def _run_vfp_sync(store_db: str, profile: Optional[str] = None):
    """Background task to run VFP sync."""
    if run_headless is None:
        logging.error("vfp_dbf_to_rdsv2 module not available")
        return

    try:
        # Use store_db as profile if not specified
        sync_profile = profile or store_db
        logging.info(f"Starting VFP sync for store {store_db} with profile {sync_profile}")
        run_headless(cfg_path=None, profile=sync_profile, auto_sync=False)
        logging.info(f"Completed VFP sync for store {store_db}")
    except Exception as e:
        logging.error(f"Error during VFP sync for store {store_db}: {e}", exc_info=True)


@router.post("/trigger")
def trigger_sync(
    store: Optional[str] = Query(
        default=None, description="Store identifier (store_id or store_db)"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_auth_user),
) -> Dict[str, Any]:
    """Trigger a VFP DBF to RDS sync operation for the authenticated user's store."""
    if run_headless is None:
        raise HTTPException(
            status_code=503,
            detail="VFP sync module not available. Please ensure vfp_dbf_to_rdsv2.py is accessible.",
        )

    store_info = _get_store_info(user, store)
    store_db = store_info.get("store_db")

    if not store_db:
        raise HTTPException(status_code=400, detail="Store database name not found")

    # Use store_db as the profile name for VFP config
    profile = store_db

    # Add sync task to background tasks
    background_tasks.add_task(_run_vfp_sync, store_db, profile)

    return {
        "status": "started",
        "message": f"VFP sync started for store {store_db}",
        "store_db": store_db,
    }
