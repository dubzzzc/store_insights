from fastapi import APIRouter, Depends, HTTPException
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/sales")
def get_sales_insights(user: dict = Depends(get_auth_user)):
    try:
        db_name = user['store_db']
        db_user = user['db_user']
        db_pass = user['db_pass']

        engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}")

        seven_days_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

        with engine.connect() as conn:
            # Step 1: Filter for valid sales (RFLAG = 0) and ensure SALE has lines 950 and 980
            sale_filter_query = text("""
                SELECT SALE
                FROM jnl
                WHERE RFLAG = 0 AND LINE IN (950, 980) AND DATE >= :seven_days_ago
                GROUP BY SALE
                HAVING COUNT(DISTINCT LINE) = 2
            """)
            valid_sales = conn.execute(sale_filter_query, {"seven_days_ago": seven_days_ago}).scalars().all()

            if not valid_sales:
                return {"store": user["email"], "sales": []}

            # Step 2: Gather item-level data from valid sales with SKU > 0
            items_query = text("""
                SELECT DATE, SALE, QTY, PACK, SKU, DESCRIPTION, PRICE
                FROM jnl
                WHERE SALE IN :valid_sales AND RFLAG = 0 AND SKU > 0 AND DATE >= :seven_days_ago
            """)
            result = conn.execute(
                items_query,
                {"valid_sales": tuple(valid_sales), "seven_days_ago": seven_days_ago}
            ).mappings()

            # Step 3: Aggregate per day
            sales_data = {}
            for row in result:
                date_key = row["DATE"].strftime('%Y-%m-%d')
                qty_sold = row["QTY"] * row["PACK"]

                if date_key not in sales_data:
                    sales_data[date_key] = {
                        "total_items_sold": 0,
                        "total_sales": 0.0
                    }

                sales_data[date_key]["total_items_sold"] += qty_sold
                sales_data[date_key]["total_sales"] += row["PRICE"]

            # Convert to sorted list (most recent first)
            sorted_sales = sorted(sales_data.items(), reverse=True)
            response = [
                {
                    "date": date,
                    "total_items_sold": int(data["total_items_sold"]),
                    "total_sales": round(data["total_sales"], 2)
                }
                for date, data in sorted_sales
            ]

            return {"store": user["email"], "sales": response[:7]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
