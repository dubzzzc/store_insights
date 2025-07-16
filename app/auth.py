from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import jwt, os, bcrypt
from app.db_core import get_core_connection
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()

SECRET = os.getenv("JWT_SECRET", "changeme")

class LoginInput(BaseModel):
    email: str
    password: str

# ✅ This is the correct login endpoint
@router.post("/login")
def login(data: LoginInput):
    print(f"🔐 Login attempt for {data.email}")
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (data.email,))
    user = cursor.fetchone()
    conn.close()

    print(f"🧠 DB User Fetched: {user}")

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    raw_input = data.password.encode()
    stored_hash = user["password_hash"].encode()

    try:
        matched = bcrypt.checkpw(raw_input, stored_hash)
    except Exception as e:
        print(f"💥 bcrypt error: {e}")
        raise HTTPException(status_code=500, detail="bcrypt failure")

    if not matched:
        raise HTTPException(status_code=401, detail="Invalid password")

    print("✅ Password matched")

    token = jwt.encode({
        "email": user["email"],
        "store_db": user["store_db"],
        "db_user": user["db_user"],
        "db_pass": user["db_pass"]
    }, SECRET, algorithm="HS256")

    return {"token": token}

# ✅ This is the token auth dependency
def get_auth_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
