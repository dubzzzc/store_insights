"""Bootstrap helpers that prepare the core database on startup."""

from __future__ import annotations

import logging
import os
from typing import Optional

import bcrypt
import mysql.connector
from mysql.connector import errorcode

from app.db_core import get_core_connection


logger = logging.getLogger(__name__)


def _has_column(cursor: mysql.connector.cursor.MySQLCursorDict, table: str, column: str) -> bool:
    """Return True when the requested column exists on the given table."""

    try:
        cursor.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (column,))
        return cursor.fetchone() is not None
    except mysql.connector.Error as err:  # pragma: no cover - defensive fallback
        if err.errno == errorcode.ER_NO_SUCH_TABLE:
            return False
        raise


def _should_reset_password() -> bool:
    return os.getenv("DEFAULT_ADMIN_RESET_PASSWORD", "false").lower() in {"1", "true", "yes", "on"}


def bootstrap_admin_user() -> None:
    """Ensure a default admin account exists for managing the platform."""

    email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@storeinsights.com")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    full_name = os.getenv("DEFAULT_ADMIN_NAME", "Store Insights Admin")

    if not password:
        logger.info("Skipping admin bootstrap because DEFAULT_ADMIN_PASSWORD is not set")
        return

    try:
        conn = get_core_connection()
    except Exception as exc:  # pragma: no cover - startup guard
        logger.warning("Unable to connect to core database for admin bootstrap: %s", exc)
        return

    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SHOW TABLES LIKE 'users'")
        if cursor.fetchone() is None:
            logger.info("Users table not present; skipping admin bootstrap")
            return

        supports_full_name = _has_column(cursor, "users", "full_name")
        supports_user_role = _has_column(cursor, "users", "user_role")

        cursor.execute("SELECT id, user_role FROM users WHERE email = %s", (email,))
        existing: Optional[dict] = cursor.fetchone()

        if existing:
            updates = []
            params = []

            if supports_user_role and existing.get("user_role") != "admin":
                updates.append("user_role = %s")
                params.append("admin")

            if _should_reset_password():
                hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                updates.append("password_hash = %s")
                params.append(hashed)

            if updates:
                params.append(existing["id"])
                cursor.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
                    tuple(params),
                )
                conn.commit()
                logger.info("Updated default admin account")
            else:
                logger.debug("Default admin already present; no changes required")

            return

        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        columns = ["email", "password_hash"]
        values = [email, hashed]

        if supports_full_name:
            columns.append("full_name")
            values.append(full_name)

        if supports_user_role:
            columns.append("user_role")
            values.append("admin")

        placeholders = ", ".join(["%s"] * len(values))
        column_clause = ", ".join(columns)

        cursor.execute(
            f"INSERT INTO users ({column_clause}) VALUES ({placeholders})",
            tuple(values),
        )
        conn.commit()
        logger.info("Created default admin account for %s", email)
    except mysql.connector.Error as exc:  # pragma: no cover - startup guard
        conn.rollback()
        logger.error("Failed to bootstrap admin user: %s", exc)
    finally:
        cursor.close()
        conn.close()
