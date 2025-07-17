from fastapi import APIRouter, Depends, HTTPException
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/sales")
def get_sales_insights(user: dict = Depends(get_auth_user)):
    try:
        # Extract user credentials
        db_name = user['store_db']
        db_user = user['db_user']
        db_pass = user['db_pass']

        print(f"📍 Authenticated as: {user['email']}")
        print(f"🔑 Connecting to DB: {db_name} with user {db_user}")

        engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}")

        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        print(f"📅 Filtering from: {seven_days_ago}")

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT SALE
                FROM jnl
                WHERE RFLAG = 0 AND DATE >= :seven_days_ago
                GROUP BY SALE
                HAVING SUM(LINE = 950) > 0 AND SUM(LINE = 980) > 0
            """), {"seven_days_ago": seven_days_ago}).mappings()

            valid_sales = [row["SALE"] for row in result]
            print(f"✅ Valid sales with tender lines: {len(valid_sales)} found")

            if not valid_sales:
                return {"store": user["email"], "sales": []}

            items = conn.execute(text("""
                SELECT DATE, SKU, DESCRIPTION, PACK, QTY, PRICE
                FROM jnl
                WHERE SALE IN :sales AND SKU > 0
            """), {"sales": tuple(valid_sales)}).mappings()

            sales_summary = {}

            for row in items:
                date_str = row["DATE"].strftime('%Y-%m-%d')
                qty = row["PACK"] * row["QTY"]

                if date_str not in sales_summary:
                    sales_summary[date_str] = {
                        "total_items_sold": 0,
                        "total_sales": 0.0,
                    }

                sales_summary[date_str]["total_items_sold"] += qty
                sales_summary[date_str]["total_sales"] += row["PRICE"]

                print(f"🧾 {date_str} | SKU: {row['SKU']} | {row['DESCRIPTION']} | Qty: {qty} | Price: {row['PRICE']}")

            # Format response
            return {
                "store": user["email"],
                "sales": [
                    {
                        "date": date,
                        "total_items_sold": summary["total_items_sold"],
                        "total_sales": round(summary["total_sales"], 2)
                    }
                    for date, summary in sorted(sales_summary.items(), reverse=True)
                ]
            }

    except Exception as e:
        print(f"💥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
