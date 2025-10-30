from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

router = APIRouter()

def _select_store(user: dict, requested_store: Optional[str]) -> Dict[str, Any]:
    stores: List[Dict[str, Any]] = user.get("stores") or []

    if not stores and user.get("store_db"):
        stores = [
            {
                "store_db": user.get("store_db"),
                "db_user": user.get("db_user"),
                "db_pass": user.get("db_pass"),
            }
        ]

    if not stores:
        raise HTTPException(status_code=403, detail="No stores associated with this account")

    if requested_store:
        for store in stores:
            store_ids = [
                str(store.get("store_id")) if store.get("store_id") is not None else None,
                store.get("store_db"),
            ]
            if requested_store in filter(None, store_ids):
                return store
        raise HTTPException(status_code=404, detail="Requested store is not assigned to this user")

    if len(stores) > 1:
        raise HTTPException(status_code=400, detail="Multiple stores available. Specify the 'store' query parameter.")

    return stores[0]


@router.get("/sales")
def get_sales_insights(
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    user: dict = Depends(get_auth_user),
):
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

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
            """), {"seven_days_ago": seven_days_ago}).mappings()

            valid_sales = [row["SALE"] for row in result]
            print(f"✅ Valid sales with tender lines: {len(valid_sales)} found")

            if not valid_sales:
                return {"store": {"store_db": db_name, "store_id": selected_store.get("store_id")}, "sales": []}

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
                "store": {
                    "store_db": db_name,
                    "store_id": selected_store.get("store_id"),
                    "store_name": selected_store.get("store_name"),
                },
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
