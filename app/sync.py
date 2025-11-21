from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import logging

import mysql.connector
from mysql.connector import errorcode
from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_auth_user
from app.db_core import get_core_connection

router = APIRouter()

# Note: VFP sync now runs locally via local_sync_agent.py
# The server only manages sync requests in the database


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


