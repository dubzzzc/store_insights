from enum import Enum
from typing import List, Optional

import bcrypt
import os
from fastapi import APIRouter, Depends, Header, HTTPException, status
import mysql.connector
from mysql.connector import errorcode
from pydantic import BaseModel, EmailStr, Field

from app.db_core import get_core_connection

router = APIRouter()


def _require_admin(api_key: str = Header(..., alias="X-Admin-Key")) -> None:
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")


def _has_column(cursor, table_name: str, column_name: str) -> bool:
    """Return True when the given column exists on the target table."""

    try:
        cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE %s", (column_name,))
        return cursor.fetchone() is not None
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_NO_SUCH_TABLE:
            return False
        raise


class UserRole(str, Enum):
    admin = "admin"
    owner = "owner"
    user = "user"


class StoreAssignment(BaseModel):
    store_db: str = Field(..., description="Database name for the tenant store")
    db_user: str = Field(..., description="Database user with read access")
    db_pass: str = Field(..., description="Database password for the store user")
    store_name: Optional[str] = Field(None, description="Friendly name for the store")


class StoreAssignmentRecord(StoreAssignment):
    id: Optional[int] = Field(None, description="Identifier from the user_stores table when available")


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, description="Optional display name for the user")
    user_role: Optional[UserRole] = Field(
        None,
        description="Role of the user (admin, owner, or user). Defaults to owner when supported.",
    )
    stores: List[StoreAssignment] = Field(default_factory=list)


class CreateUserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]
    user_role: Optional[UserRole]
    stores: List[StoreAssignment]


class AdminUser(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]
    user_role: Optional[UserRole]
    stores: List[StoreAssignmentRecord] = Field(default_factory=list)


class AdminUsersResponse(BaseModel):
    users: List[AdminUser]


# Response container for store assignment actions
class StoreAssignmentResponse(BaseModel):
    store: StoreAssignmentRecord


# --- New admin helpers for uploader creds management ---
class UpdateStoreCredsRequest(BaseModel):
    store_db: Optional[str] = None
    db_user: Optional[str] = None
    db_pass: Optional[str] = None
    store_name: Optional[str] = None


class RotateApiKeyResponse(BaseModel):
    store_db: str
    api_key: str


class ProvisionStoreRequest(BaseModel):
    store_db: str = Field(..., description="New or existing DB name to ensure")
    db_user: str = Field(..., description="Database user to (create and) grant access")
    db_pass: str = Field(..., description="Password for the database user")
    store_name: Optional[str] = Field(None, description="Optional label; not persisted here")


class ProvisionStoreResponse(BaseModel):
    success: bool
    created_database: bool
    ensured_user: bool
    granted_privileges: bool
    message: Optional[str] = None


def _validate_identifier(value: str, kind: str) -> None:
    import re
    if not value or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise HTTPException(status_code=400, detail=f"Invalid {kind}. Use letters, numbers, underscore only.")


@router.post("/stores/provision", response_model=ProvisionStoreResponse)
def provision_store(payload: ProvisionStoreRequest, _: None = Depends(_require_admin)):
    """Create the tenant database if missing, ensure user exists, and grant SELECT privileges.

    Requires env: STORE_RDS_HOST, STORE_RDS_ADMIN_USER, STORE_RDS_ADMIN_PASS, optional STORE_RDS_PORT.
    """
    host = os.getenv("STORE_RDS_HOST")
    admin_user = os.getenv("STORE_RDS_ADMIN_USER")
    admin_pass = os.getenv("STORE_RDS_ADMIN_PASS")
    port = int(os.getenv("STORE_RDS_PORT", "3306"))
    if not host or not admin_user or not admin_pass:
        raise HTTPException(status_code=500, detail="Server missing STORE_RDS_* admin configuration")

    # Basic validation to avoid SQL injection on identifiers
    _validate_identifier(payload.store_db, "store_db")
    _validate_identifier(payload.db_user, "db_user")

    created_db = False
    ensured_user = False
    granted = False

    try:
        admin_conn = mysql.connector.connect(host=host, port=port, user=admin_user, password=admin_pass)
        admin_cur = admin_conn.cursor()
        try:
            # Create DB if missing
            admin_cur.execute(f"CREATE DATABASE IF NOT EXISTS `{payload.store_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            admin_conn.commit()
            # Detect if it was created: query information_schema
            admin_cur.execute(
                """
                SELECT SCHEMA_NAME FROM information_schema.schemata WHERE SCHEMA_NAME = %s
                """,
                (payload.store_db,),
            )
            if admin_cur.fetchone():
                created_db = True  # indicates now exists; not strictly new vs existing

            # Ensure user exists
            try:
                admin_cur.execute("CREATE USER IF NOT EXISTS %s@'%%' IDENTIFIED BY %s", (payload.db_user, payload.db_pass))
                ensured_user = True
            except mysql.connector.Error as err:
                # Older MySQL may not support IF NOT EXISTS; try create and ignore duplicate
                if err.errno == errorcode.ER_PARSE_ERROR:
                    try:
                        admin_cur.execute("CREATE USER %s@'%%' IDENTIFIED BY %s", (payload.db_user, payload.db_pass))
                        ensured_user = True
                    except mysql.connector.Error as err2:
                        if err2.errno == errorcode.ER_CANNOT_USER:  # e.g., user exists
                            ensured_user = True
                        else:
                            raise
                elif err.errno == errorcode.ER_CANNOT_USER:
                    ensured_user = True
                else:
                    raise

            # Grant privileges
            # Use two-step to avoid IDENTIFIED BY in GRANT (deprecated)
            admin_cur.execute(f"GRANT SELECT ON `{payload.store_db}`.* TO %s@'%'", (payload.db_user,))
            admin_cur.execute("FLUSH PRIVILEGES")
            admin_conn.commit()
            granted = True
        finally:
            try:
                admin_cur.close(); admin_conn.close()
            except Exception:
                pass
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Provisioning failed: {err}")

    return ProvisionStoreResponse(
        success=True,
        created_database=created_db,
        ensured_user=ensured_user,
        granted_privileges=granted,
        message="Provisioned successfully",
    )

@router.post("/users", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: CreateUserRequest, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")
        supports_user_role = _has_column(cursor, "users", "user_role")

        cursor.execute("SELECT id FROM users WHERE email = %s", (payload.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

        password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        insert_columns = ["email", "password_hash"]
        insert_values = [payload.email, password_hash]

        stored_full_name: Optional[str] = None
        stored_user_role: Optional[str] = None
        if payload.full_name and supports_full_name:
            insert_columns.append("full_name")
            insert_values.append(payload.full_name)
            stored_full_name = payload.full_name

        if supports_user_role:
            role_value = payload.user_role.value if payload.user_role else UserRole.owner.value
            insert_columns.append("user_role")
            insert_values.append(role_value)
            stored_user_role = role_value

        placeholders = ", ".join(["%s"] * len(insert_values))
        column_clause = ", ".join(insert_columns)

        cursor.execute(
            f"INSERT INTO users ({column_clause}) VALUES ({placeholders})",
            tuple(insert_values),
        )
        user_id = cursor.lastrowid

        legacy_store_written = False
        for store in payload.stores:
            try:
                cursor.execute(
                    """
                    INSERT INTO user_stores (user_id, store_db, db_user, db_pass, store_name)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        store.store_db,
                        store.db_user,
                        store.db_pass,
                        store.store_name,
                    ),
                )
            except mysql.connector.Error:
                if legacy_store_written or len(payload.stores) != 1:
                    raise

                cursor.execute(
                    """
                    UPDATE users
                    SET store_db = %s, db_user = %s, db_pass = %s, store_name = %s
                    WHERE id = %s
                    """,
                    (
                        store.store_db,
                        store.db_user,
                        store.db_pass,
                        store.store_name,
                        user_id,
                    ),
                )
                legacy_store_written = True

        conn.commit()

        return CreateUserResponse(
            id=user_id,
            email=payload.email,
            full_name=stored_full_name,
            user_role=stored_user_role,
            stores=payload.stores,
        )
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {exc}")
    finally:
        cursor.close()
        conn.close()


# --- New endpoints: rotate API key and update creds ---
def _ensure_api_key_column(cursor) -> None:
    if not _has_column(cursor, "user_stores", "api_key"):
        # Best-effort: add column if missing
        try:
            cursor.execute("ALTER TABLE user_stores ADD COLUMN api_key VARCHAR(255) NULL")
        except mysql.connector.Error:
            pass


def _random_key(length: int = 48) -> str:
    import secrets, string
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@router.post("/stores/{store_db}/api-key/rotate", response_model=RotateApiKeyResponse)
def rotate_store_api_key(store_db: str, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        _ensure_api_key_column(cursor)
        new_key = _random_key()
        cursor.execute(
            "UPDATE user_stores SET api_key = %s WHERE store_db = %s",
            (new_key, store_db),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Store not found in user_stores")
        conn.commit()
        return RotateApiKeyResponse(store_db=store_db, api_key=new_key)
    except HTTPException:
        conn.rollback(); raise
    except mysql.connector.Error as err:
        conn.rollback(); raise HTTPException(status_code=500, detail=f"Failed to rotate api key: {err}")
    finally:
        cursor.close(); conn.close()


@router.put("/stores/{store_db}/creds", response_model=StoreAssignmentResponse)
def update_store_creds(store_db: str, payload: UpdateStoreCredsRequest, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, store_db, db_user, db_pass, store_name FROM user_stores WHERE store_db = %s",
            (store_db,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Store not found in user_stores")

        update_cols = []
        update_vals = []
        if payload.store_db:
            update_cols.append("store_db = %s"); update_vals.append(payload.store_db)
        if payload.db_user:
            update_cols.append("db_user = %s"); update_vals.append(payload.db_user)
        if payload.db_pass:
            update_cols.append("db_pass = %s"); update_vals.append(payload.db_pass)
        if payload.store_name is not None:
            update_cols.append("store_name = %s"); update_vals.append(payload.store_name)

        if update_cols:
            update_vals.append(store_db)
            cursor.execute(
                f"UPDATE user_stores SET {', '.join(update_cols)} WHERE store_db = %s",
                tuple(update_vals),
            )
            conn.commit()

        cursor.execute(
            "SELECT id, store_db, db_user, db_pass, store_name FROM user_stores WHERE store_db = %s",
            (payload.store_db or store_db,),
        )
        out = cursor.fetchone()
        return StoreAssignmentResponse(
            store=StoreAssignmentRecord(
                id=out["id"],
                store_db=out["store_db"],
                db_user=out["db_user"],
                db_pass=out["db_pass"],
                store_name=out.get("store_name"),
            )
        )
    except HTTPException:
        conn.rollback(); raise
    except mysql.connector.Error as err:
        conn.rollback(); raise HTTPException(status_code=500, detail=f"Failed to update creds: {err}")
    finally:
        cursor.close(); conn.close()

def _ensure_user_exists(cursor, user_id: int) -> None:
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


def _list_store_assignments(cursor, user_id: int) -> List[StoreAssignmentRecord]:
    stores: List[StoreAssignmentRecord] = []

    try:
        cursor.execute(
            """
            SELECT id, store_db, db_user, db_pass, store_name
            FROM user_stores
            WHERE user_id = %s
            ORDER BY id
            """,
            (user_id,),
        )
        for row in cursor.fetchall():
            stores.append(
                StoreAssignmentRecord(
                    id=row.get("id"),
                    store_db=row.get("store_db"),
                    db_user=row.get("db_user"),
                    db_pass=row.get("db_pass"),
                    store_name=row.get("store_name"),
                )
            )
    except mysql.connector.Error as err:
        if err.errno != errorcode.ER_NO_SUCH_TABLE:
            raise

    if not stores:
        cursor.execute(
            "SELECT store_db, db_user, db_pass, store_name FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        if row and row.get("store_db"):
            stores.append(
                StoreAssignmentRecord(
                    store_db=row.get("store_db"),
                    db_user=row.get("db_user"),
                    db_pass=row.get("db_pass"),
                    store_name=row.get("store_name"),
                )
            )

    return stores


@router.get("/users", response_model=AdminUsersResponse)
def list_users(_: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")
        supports_user_role = _has_column(cursor, "users", "user_role")

        selected_columns = ["id", "email"]
        if supports_full_name:
            selected_columns.append("full_name")
        if supports_user_role:
            selected_columns.append("user_role")

        column_clause = ", ".join(selected_columns)
        cursor.execute(f"SELECT {column_clause} FROM users ORDER BY email")
        rows = cursor.fetchall() or []

        users: List[AdminUser] = []
        for row in rows:
            stores = _list_store_assignments(cursor, row["id"])
            user_role = row.get("user_role") if supports_user_role else None
            users.append(
                AdminUser(
                    id=row["id"],
                    email=row["email"],
                    full_name=row.get("full_name"),
                    user_role=user_role,
                    stores=stores,
                )
            )

        return AdminUsersResponse(users=users)
    except HTTPException:
        raise
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Failed to load users: {err}")
    finally:
        cursor.close()
        conn.close()


@router.get("/users/{user_id}", response_model=AdminUser)
def get_user(user_id: int, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")
        supports_user_role = _has_column(cursor, "users", "user_role")

        selected_columns = ["id", "email"]
        if supports_full_name:
            selected_columns.append("full_name")
        if supports_user_role:
            selected_columns.append("user_role")

        column_clause = ", ".join(selected_columns)
        cursor.execute(
            f"SELECT {column_clause} FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        stores = _list_store_assignments(cursor, user_id)

        user_role = row.get("user_role") if supports_user_role else None

        return AdminUser(
            id=row["id"],
            email=row["email"],
            full_name=row.get("full_name"),
            user_role=user_role,
            stores=stores,
        )
    except HTTPException:
        raise
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Failed to load user: {err}")
    finally:
        cursor.close()
        conn.close()


## duplicate removed: StoreAssignmentResponse defined above


@router.post(
    "/users/{user_id}/stores",
    response_model=StoreAssignmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_store_to_user(user_id: int, payload: StoreAssignment, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        _ensure_user_exists(cursor, user_id)

        try:
            cursor.execute(
                "SELECT id FROM user_stores WHERE user_id = %s AND store_db = %s",
                (user_id, payload.store_db),
            )
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise
        else:
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Store already assigned to this user",
                )

        record_id: Optional[int] = None

        try:
            cursor.execute(
                """
                INSERT INTO user_stores (user_id, store_db, db_user, db_pass, store_name)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    payload.store_db,
                    payload.db_user,
                    payload.db_pass,
                    payload.store_name,
                ),
            )
            record_id = cursor.lastrowid
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise

            cursor.execute(
                "SELECT store_db FROM users WHERE id = %s",
                (user_id,),
            )
            existing = cursor.fetchone()
            if existing and existing.get("store_db"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User already has a store assignment",
                )

            cursor.execute(
                """
                UPDATE users
                SET store_db = %s, db_user = %s, db_pass = %s, store_name = %s
                WHERE id = %s
                """,
                (
                    payload.store_db,
                    payload.db_user,
                    payload.db_pass,
                    payload.store_name,
                    user_id,
                ),
            )

        conn.commit()

        return StoreAssignmentResponse(
            store=StoreAssignmentRecord(id=record_id, **payload.dict()),
        )
    except HTTPException:
        conn.rollback()
        raise
    except mysql.connector.Error as err:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to assign store: {err}")
    finally:
        cursor.close()
        conn.close()


class RemoveStoreResponse(BaseModel):
    success: bool
    stores: List[StoreAssignmentRecord]


@router.delete("/users/{user_id}/stores/{store_identifier}", response_model=RemoveStoreResponse)
def remove_store_from_user(user_id: int, store_identifier: str, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        _ensure_user_exists(cursor, user_id)

        deleted = False

        try:
            try:
                store_id = int(store_identifier)
                cursor.execute(
                    "DELETE FROM user_stores WHERE user_id = %s AND id = %s",
                    (user_id, store_id),
                )
            except ValueError:
                cursor.execute(
                    "DELETE FROM user_stores WHERE user_id = %s AND store_db = %s",
                    (user_id, store_identifier),
                )

            if cursor.rowcount:
                deleted = True
                conn.commit()
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise

            cursor.execute(
                "SELECT store_db FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            current_db = row.get("store_db") if row else None
            if not current_db:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Store assignment not found for user",
                )

            if store_identifier not in ("legacy", current_db):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Store assignment not found for user",
                )

            cursor.execute(
                """
                UPDATE users
                SET store_db = NULL, db_user = NULL, db_pass = NULL, store_name = NULL
                WHERE id = %s
                """,
                (user_id,),
            )
            conn.commit()
            deleted = True

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Store assignment not found for user",
            )

        stores = _list_store_assignments(cursor, user_id)

        return RemoveStoreResponse(success=True, stores=stores)
    except HTTPException:
        conn.rollback()
        raise
    except mysql.connector.Error as err:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove store: {err}")
    finally:
        cursor.close()
        conn.close()
