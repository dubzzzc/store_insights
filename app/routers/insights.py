from fastapi import APIRouter, Depends, HTTPException
from app.auth.auth import get_auth_user
from sqlalchemy import create_engine, text
from collections import defaultdict
from datetime import datetime

router = APIRouter()

@router.get("/insights/sales")
def get_sales_insights(user: dict = Depends(get_auth_user)):
    try:
        # Pull credentials from token
        db_name = user['store_db']
        db_user = user['db_user']
        db_pass = user['db_pass']

        # Create engine dynamically per user
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}")

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    DATE(sale_date) as date, 
                    SUM(qty) as total_items_sold,
                    SUM(amount) as total_sales
                FROM sales
                GROUP BY DATE(sale_date)
                ORDER BY DATE(sale_date) DESC
                LIMIT 7
            """))

            sales_data = []
            for row in result:
                sales_data.append({
                    "date": row["date"].strftime('%Y-%m-%d'),
                    "total_items_sold": int(row["total_items_sold"]),
                    "total_sales": float(row["total_sales"])
                })

            return {"store": user["email"], "sales": sales_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
