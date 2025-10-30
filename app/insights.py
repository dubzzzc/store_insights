from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import create_engine, text, bindparam

from app.auth import get_auth_user

router = APIRouter()

def _select_store(user: dict, requested_store: Optional[str]) -> Dict[str, Any]:
    stores: List[Dict[str, Any]] = user.get("stores") or []

    # Support legacy tokens that only carried a single store at the top level
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

        engine = create_engine(
            f"mysql+pymysql://{db_user}:{db_pass}@spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}"
        )

        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        with engine.connect() as conn:
            sale_numbers = conn.execute(
                text(
                    """
                    SELECT SALE
                    FROM jnl
                    WHERE RFLAG = 0
                      AND DATE >= :seven_days_ago
                    GROUP BY SALE
                    """
                ),
                {"seven_days_ago": seven_days_ago},
            ).scalars().all()

            if not sale_numbers:
                response_meta = {
                    "store_db": db_name,
                    "store_id": selected_store.get("store_id"),
                    "store_name": selected_store.get("store_name"),
                }
                return {"store": response_meta, "sales": []}

            items_query = (
                text(
                    """
                    SELECT DATE, SKU, DESCRIPTION, PACK, QTY, PRICE
                    FROM jnl
                    WHERE SALE IN :sales
                      AND SKU > 0
                    """
                )
                .bindparams(bindparam("sales", expanding=True))
            )

            line_items = conn.execute(items_query, {"sales": tuple(sale_numbers)}).mappings()

            sales_summary: Dict[str, Dict[str, float]] = {}

            for row in line_items:
                sale_date = row["DATE"]
                if sale_date is None:
                    continue

                date_key = sale_date.strftime("%Y-%m-%d")
                pack = row.get("PACK") or 1
                qty = row.get("QTY") or 0
                line_qty = pack * qty

                price = row.get("PRICE") or 0
                if isinstance(price, Decimal):
                    price = float(price)

                if date_key not in sales_summary:
                    sales_summary[date_key] = {
                        "total_items_sold": 0,
                        "total_sales": 0.0,
                    }

                sales_summary[date_key]["total_items_sold"] += line_qty
                sales_summary[date_key]["total_sales"] += price

            ordered_days = sorted(sales_summary.items(), reverse=True)
            sales_data = [
                {
                    "date": date,
                    "total_items_sold": int(summary["total_items_sold"]),
                    "total_sales": round(float(summary["total_sales"]), 2),
                }
                for date, summary in ordered_days[:7]
            ]

            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            return {"store": response_meta, "sales": sales_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
