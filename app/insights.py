from fastapi import APIRouter, Depends, HTTPException
import jwt
from jwt import InvalidTokenError
from app.db_router import get_store_connection
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "changeme")

def get_auth_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET, algorithms=["HS256"])
        return payload
    except InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")


@router.get("/daily_sales")
def daily_sales(user=Depends(get_auth_user)):
    conn = get_store_connection(user['store_db'], user['db_user'], user['db_pass'])
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT DATE(sale_date) as sale_day, SUM(amount) as total_sales
        FROM sales
        GROUP BY sale_day
        ORDER BY sale_day DESC
        LIMIT 30;
    """)
    results = cursor.fetchall()
    conn.close()
    return results
