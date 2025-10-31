from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any, List

"""
This module re-exports the sales insights router that dynamically adapts to
Visual FoxPro schema differences (for example, DESCRIPTION vs DESCRIPT). The
main application continues to import the router from ``app.insights`` so we keep
that import path stable.
"""

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
              """)).mappings()

            sales_data = []
            for row in result:
                sales_data.append({
                    "date": row["date"].strftime('%Y-%m-%d'),
                    "total_items_sold": int(row["total_items_sold"]),
                    "total_sales": float(row["total_sales"])
                })

            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            return {"store": response_meta, "sales": sales_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
