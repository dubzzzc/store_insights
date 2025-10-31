from decimal import Decimal
from datetime import date as date_type, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import bindparam, create_engine, text

from app.auth import get_auth_user

router = APIRouter()


_COLUMN_SPECS: Dict[str, Dict[str, Any]] = {
    "sale": {"candidates": ["sale"], "required": True},
    "rflag": {"candidates": ["rflag"], "required": False},
    "date": {"candidates": ["date", "sale_date", "trandate", "trans_date"], "required": True},
    "sku": {"candidates": ["sku", "item", "itemno", "upc"], "required": True},
    "description": {
        "candidates": [
            "description",
            "descr",
            "descript",
            "item_desc",
            "desc",
            "itemdescription",
            "product_description",
        ],
        "required": False,
    },
    "pack": {"candidates": ["pack", "package", "packsize", "pack_size"], "required": False},
    "qty": {"candidates": ["qty", "quantity", "qtysold", "qty_sold"], "required": True},
    "price": {"candidates": ["price", "amount", "total", "extended", "extprice", "extendedprice"], "required": True},
}


def _quote_identifier(column: str) -> str:
    if column.startswith("`") and column.endswith("`"):
        return column
    return f"`{column}`"


def _resolve_jnl_columns(connection, schema: str) -> Dict[str, Optional[str]]:
    result = connection.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            """
        ),
        {"schema": schema, "table": "jnl"},
    )
    available = {row[0].lower(): row[0] for row in result}

    resolved: Dict[str, Optional[str]] = {}
    missing: List[str] = []

    for key, spec in _COLUMN_SPECS.items():
        column_name: Optional[str] = None
        for candidate in spec["candidates"]:
            if candidate.lower() in available:
                column_name = available[candidate.lower()]
                break

        if column_name is None and spec.get("required"):
            missing.append(spec["candidates"][0])

        resolved[key] = column_name

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"jnl table is missing required columns: {', '.join(sorted(missing))}",
        )

    return resolved


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
            columns = _resolve_jnl_columns(conn, db_name)

            sale_column = _quote_identifier(columns["sale"])
            date_column = _quote_identifier(columns["date"])
            sku_column = _quote_identifier(columns["sku"])
            qty_column = _quote_identifier(columns["qty"])
            price_column = _quote_identifier(columns["price"])
            rflag_column = _quote_identifier(columns["rflag"]) if columns.get("rflag") else None
            description_column = (
                _quote_identifier(columns["description"]) if columns.get("description") else "''"
            )
            pack_column = _quote_identifier(columns["pack"]) if columns.get("pack") else None

            where_clauses = [f"{date_column} >= :seven_days_ago"]
            if rflag_column:
                where_clauses.append(f"{rflag_column} = 0")

            sales_sql = f"""
                SELECT {sale_column} AS sale_id
                FROM jnl
                WHERE {' AND '.join(where_clauses)}
                GROUP BY {sale_column}
            """

            result = conn.execute(text(sales_sql), {"seven_days_ago": seven_days_ago}).mappings()

            valid_sales = [row["sale_id"] for row in result]
            print(f"✅ Valid sales with tender lines: {len(valid_sales)} found")

            if not valid_sales:
                return {
                    "store": {
                        "store_db": db_name,
                        "store_id": selected_store.get("store_id"),
                        "store_name": selected_store.get("store_name"),
                    },
                    "sales": [],
                    "summary": {
                        "gross_sales": 0.0,
                        "total_items": 0.0,
                        "days_captured": 0,
                        "average_daily_sales": 0.0,
                    },
                    "top_items": [],
                }

            select_fields = [
                f"{date_column} AS sale_date",
                f"{sku_column} AS sku",
                f"{qty_column} AS quantity",
                f"{price_column} AS price",
            ]

            if description_column == "''":
                select_fields.append("'' AS description")
            else:
                select_fields.append(f"{description_column} AS description")

            if pack_column:
                select_fields.append(f"{pack_column} AS pack")
            else:
                select_fields.append("1 AS pack")

            items_sql = text(
                f"""
                SELECT {', '.join(select_fields)}
                FROM jnl
                WHERE {sale_column} IN :sales AND {sku_column} > 0
                """
            ).bindparams(bindparam("sales", expanding=True))

            items = conn.execute(items_sql, {"sales": tuple(valid_sales)}).mappings()

            sales_summary: Dict[str, Dict[str, float]] = {}
            sku_summary: Dict[str, Dict[str, Any]] = {}
            total_sales_value = 0.0
            total_items_value = 0.0

            for row in items:
                date_value = row.get("sale_date")
                if isinstance(date_value, datetime):
                    date_str = date_value.strftime('%Y-%m-%d')
                elif isinstance(date_value, date_type):
                    date_str = date_value.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_value)

                pack_value = _coerce_number(row.get("pack"), default=1.0)
                qty_value = _coerce_number(row.get("quantity"), default=0.0)
                price_value = _coerce_number(row.get("price"), default=0.0)
                qty = pack_value * qty_value

                if date_str not in sales_summary:
                    sales_summary[date_str] = {
                        "total_items_sold": 0,
                        "total_sales": 0.0,
                    }

                sales_summary[date_str]["total_items_sold"] += qty
                sales_summary[date_str]["total_sales"] += price_value

                total_items_value += qty
                total_sales_value += price_value

                sku_key_raw = row.get("sku")
                sku_key = str(sku_key_raw) if sku_key_raw is not None else ""
                if sku_key:
                    sku_entry = sku_summary.setdefault(
                        sku_key,
                        {
                            "sku": sku_key,
                            "description": "",
                            "total_items_sold": 0.0,
                            "total_sales": 0.0,
                        },
                    )
                    sku_entry["total_items_sold"] += qty
                    sku_entry["total_sales"] += price_value

                    if not sku_entry["description"]:
                        description_value = row.get("description")
                        if description_value is not None:
                            sku_entry["description"] = str(description_value)

                description_value = row.get("description")
                print(
                    "🧾 "
                    f"{date_str} | SKU: {row.get('sku')} | "
                    f"{description_value if description_value is not None else ''} | "
                    f"Qty: {qty} | Price: {price_value}"
                )

            # Format response
            days_captured = len(sales_summary)
            average_daily_sales = round(total_sales_value / days_captured, 2) if days_captured else 0.0

            top_items = sorted(
                (
                    {
                        "sku": item["sku"],
                        "description": item["description"],
                        "total_items_sold": round(item["total_items_sold"], 2),
                        "total_sales": round(item["total_sales"], 2),
                    }
                    for item in sku_summary.values()
                ),
                key=lambda item: item["total_sales"],
                reverse=True,
            )[:10]

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
                        "total_sales": round(summary["total_sales"], 2),
                    }
                    for date, summary in sorted(sales_summary.items(), reverse=True)
                ],
                "summary": {
                    "gross_sales": round(total_sales_value, 2),
                    "total_items": round(total_items_value, 2),
                    "days_captured": days_captured,
                    "average_daily_sales": average_daily_sales,
                },
                "top_items": top_items,
            }

    except Exception as e:
        print(f"💥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
