from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import jwt, os, bcrypt
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import mysql.connector
from mysql.connector import errorcode

from app.db_core import get_core_connection
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

print("✅ auth.py loaded and router defined")


router = APIRouter()
security = HTTPBearer()

SECRET = os.getenv("JWT_SECRET", "changeme")


def _has_column(cursor, table_name: str, column_name: str) -> bool:
    """Return True when the given column exists on the target table."""
    try:
        cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE %s", (column_name,))
        return cursor.fetchone() is not None
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_NO_SUCH_TABLE:
            return False
        raise


def _has_table(cursor, table_name: str) -> bool:
    """Return True when the given table exists."""
    try:
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        return cursor.fetchone() is not None
    except mysql.connector.Error:
        return False


def _ensure_login_logs_table(cursor) -> None:
    """Create login_logs table if it doesn't exist."""
    if _has_table(cursor, "login_logs"):
        return
    
    cursor.execute("""
        CREATE TABLE login_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            email VARCHAR(255) NULL,
            ip_address VARCHAR(45) NOT NULL,
            location VARCHAR(255) NULL,
            country VARCHAR(100) NULL,
            city VARCHAR(100) NULL,
            login_status ENUM('success', 'failed') NOT NULL,
            failure_reason VARCHAR(255) NULL,
            logged_in_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_id (user_id),
            INDEX idx_logged_in_at (logged_in_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)


def _ensure_last_login_column(cursor) -> None:
    """Add last_login column to users table if it doesn't exist."""
    if not _has_table(cursor, "users"):
        return
    
    if not _has_column(cursor, "users", "last_login"):
        cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME NULL")


def _get_client_ip(request: Request) -> str:
    """Extract client IP address from request, handling proxies."""
    # Check X-Forwarded-For header (for proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        ip = forwarded_for.split(",")[0].strip()
        if ip:
            return ip
    
    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to client host
    if request.client and request.client.host:
        return request.client.host
    
    return "unknown"


def _get_location_from_ip(ip_address: str) -> Dict[str, Optional[str]]:
    """Get location information from IP address using free geolocation API.
    
    Returns dict with keys: city, country, location (formatted string)
    Returns None values if API is unavailable or fails.
    """
    if not ip_address or ip_address == "unknown" or ip_address.startswith("127.") or ip_address.startswith("::1"):
        return {"city": None, "country": None, "location": None}
    
    try:
        # Using ip-api.com free tier (no API key required, 45 requests/minute limit)
        # Alternative: ipapi.co (requires API key for better rate limits)
        response = requests.get(
            f"http://ip-api.com/json/{ip_address}",
            timeout=3,
            params={"fields": "status,message,country,city"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                city = data.get("city", "")
                country = data.get("country", "")
                location = f"{city}, {country}".strip(", ") if city or country else None
                return {
                    "city": city if city else None,
                    "country": country if country else None,
                    "location": location
                }
    except Exception as e:
        print(f"⚠️ Geolocation API error for IP {ip_address}: {e}")
    
    return {"city": None, "country": None, "location": None}


def _log_login_attempt(
    cursor,
    user_id: Optional[int],
    email: str,
    ip_address: str,
    location_info: Dict[str, Optional[str]],
    login_status: str,
    failure_reason: Optional[str] = None
) -> None:
    """Log a login attempt to the login_logs table."""
    try:
        _ensure_login_logs_table(cursor)
        
        cursor.execute("""
            INSERT INTO login_logs 
            (user_id, email, ip_address, location, country, city, login_status, failure_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            email,
            ip_address,
            location_info.get("location"),
            location_info.get("country"),
            location_info.get("city"),
            login_status,
            failure_reason
        ))
    except Exception as e:
        # Don't fail login if logging fails
        print(f"⚠️ Failed to log login attempt: {e}")


def _update_last_login(cursor, user_id: int) -> None:
    """Update the last_login timestamp for a user."""
    try:
        _ensure_last_login_column(cursor)
        
        cursor.execute("""
            UPDATE users 
            SET last_login = %s 
            WHERE id = %s
        """, (datetime.now(), user_id))
    except Exception as e:
        # Don't fail login if update fails
        print(f"⚠️ Failed to update last_login: {e}")


class LoginInput(BaseModel):
    email: str
    password: str

# ✅ This is the correct login endpoint
@router.post("/login")
def login(data: LoginInput, request: Request):
    print(f"🔐 Login attempt for {data.email}")
    
    # Extract IP address and get location
    ip_address = _get_client_ip(request)
    location_info = _get_location_from_ip(ip_address)
    
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Ensure tables/columns exist
        _ensure_login_logs_table(cursor)
        _ensure_last_login_column(cursor)
        
        cursor.execute("SELECT * FROM users WHERE email = %s", (data.email,))
        user = cursor.fetchone()

        print(f"🧠 DB User Fetched: {user}")

        if not user:
            # Log failed login attempt (user not found)
            _log_login_attempt(
                cursor,
                user_id=None,
                email=data.email,
                ip_address=ip_address,
                location_info=location_info,
                login_status="failed",
                failure_reason="User not found"
            )
            conn.commit()
            raise HTTPException(status_code=401, detail="User not found")

        raw_input = data.password.encode()
        stored_hash = user["password_hash"].encode()

        try:
            matched = bcrypt.checkpw(raw_input, stored_hash)
        except Exception as e:
            print(f"💥 bcrypt error: {e}")
            # Log failed login attempt (bcrypt error)
            _log_login_attempt(
                cursor,
                user_id=user["id"],
                email=data.email,
                ip_address=ip_address,
                location_info=location_info,
                login_status="failed",
                failure_reason="bcrypt failure"
            )
            conn.commit()
            raise HTTPException(status_code=500, detail="bcrypt failure")

        if not matched:
            # Log failed login attempt (invalid password)
            _log_login_attempt(
                cursor,
                user_id=user["id"],
                email=data.email,
                ip_address=ip_address,
                location_info=location_info,
                login_status="failed",
                failure_reason="Invalid password"
            )
            conn.commit()
            raise HTTPException(status_code=401, detail="Invalid password")

        print("✅ Password matched")

        stores: List[Dict[str, Any]] = []
        role = user.get("user_role") or "owner"

        try:
            cursor.execute(
                """
                SELECT
                    us.id AS store_id,
                    us.store_db,
                    us.db_user,
                    us.db_pass,
                    us.store_name
                FROM user_stores us
                WHERE us.user_id = %s
                ORDER BY us.id
                """,
                (user["id"],),
            )
            for row in cursor.fetchall():
                stores.append(
                    {
                        "store_id": row.get("store_id"),
                        "store_db": row.get("store_db"),
                        "db_user": row.get("db_user"),
                        "db_pass": row.get("db_pass"),
                        "store_name": row.get("store_name"),
                    }
                )
        except Exception as e:
            print(f"⚠️ Unable to load store mappings from user_stores: {e}")

        if not stores:
            # Fallback to legacy single-store columns if they exist
            legacy_store_db = user.get("store_db")
            if legacy_store_db:
                stores.append(
                    {
                        "store_id": None,
                        "store_db": legacy_store_db,
                        "db_user": user.get("db_user"),
                        "db_pass": user.get("db_pass"),
                        "store_name": user.get("store_name"),
                    }
                )

        if not stores and role != "admin":
            raise HTTPException(status_code=403, detail="No stores assigned to this user")

        primary_store = stores[0] if stores else {}

        token_payload = {
            "email": user["email"],
            "stores": stores,
            "role": role,
        }

        full_name = user.get("full_name")
        if full_name:
            token_payload["full_name"] = full_name

        if primary_store:
            token_payload.update(
                {
                    "store_db": primary_store.get("store_db"),
                    "db_user": primary_store.get("db_user"),
                    "db_pass": primary_store.get("db_pass"),
                }
            )

        admin_key = None
        if role == "admin":
            admin_key = os.getenv("ADMIN_API_KEY")
            if not admin_key:
                admin_key = None

        token = jwt.encode(token_payload, SECRET, algorithm="HS256")

        # Log successful login attempt and update last_login
        _log_login_attempt(
            cursor,
            user_id=user["id"],
            email=user["email"],
            ip_address=ip_address,
            location_info=location_info,
            login_status="success",
            failure_reason=None
        )
        _update_last_login(cursor, user["id"])
        conn.commit()

        response_payload = {
            "token": token,
            "stores": stores,
            "role": role,
            "email": user["email"],
            "full_name": full_name,
        }

        if admin_key:
            response_payload["admin_key"] = admin_key

        return response_payload
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"💥 Unexpected error during login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        cursor.close()
        conn.close()

# ✅ This is the token auth dependency
def get_auth_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
