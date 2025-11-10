from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, date as date_type
from typing import Optional, Dict, Any, List
from collections import OrderedDict
import os
import atexit

router = APIRouter()

# Engine cache to reuse engines per database connection string
_engine_cache: "OrderedDict[str, Any]" = OrderedDict()


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date_type):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return None


class FreshnessTracker:
    def __init__(self) -> None:
        self._latest: Optional[datetime] = None

    def track(self, *values: Any) -> None:
        for value in values:
            dt = _coerce_datetime(value)
            if dt and (self._latest is None or dt > self._latest):
                self._latest = dt

    def to_payload(self) -> Optional[Dict[str, str]]:
        if not self._latest:
            return None
        iso = self._latest.isoformat()
        display = self._latest.strftime("%Y-%m-%d %H:%M:%S")
        return {"iso": iso, "display": display}


_MAX_CACHED_ENGINES = int(os.getenv("STORE_INSIGHTS_MAX_CACHED_ENGINES", "10"))
_POOL_SIZE = int(os.getenv("STORE_INSIGHTS_POOL_SIZE", "12"))
_POOL_MAX_OVERFLOW = int(os.getenv("STORE_INSIGHTS_POOL_MAX_OVERFLOW", "18"))
_POOL_TIMEOUT = int(os.getenv("STORE_INSIGHTS_POOL_TIMEOUT", "45"))
_POOL_RECYCLE = int(os.getenv("STORE_INSIGHTS_POOL_RECYCLE", "1200"))
_CONNECT_TIMEOUT = int(os.getenv("STORE_INSIGHTS_CONNECT_TIMEOUT", "10"))


def _get_engine(db_user: str, db_pass: str, db_name: str):
    """
    Get or create a database engine with proper connection pooling.
    Reuses engines for the same connection string to avoid connection exhaustion.
    Uses tuneable pool settings (via env vars) to reduce queue timeout errors under load.
    Limits total number of cached engines to prevent connection exhaustion.
    """
    connection_string = (
        f"mysql+pymysql://{db_user}:{db_pass}@"
        f"spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}"
    )

    engine = _engine_cache.get(connection_string)
    if engine is not None:
        _engine_cache.move_to_end(connection_string)
        return engine

    if _MAX_CACHED_ENGINES > 0 and len(_engine_cache) >= _MAX_CACHED_ENGINES:
        oldest_key, oldest_engine = _engine_cache.popitem(last=False)
        try:
            oldest_engine.dispose()
        except Exception:
            pass

    engine = create_engine(
        connection_string,
        pool_size=_POOL_SIZE,
        max_overflow=_POOL_MAX_OVERFLOW,
        pool_recycle=_POOL_RECYCLE,
        pool_pre_ping=True,
        pool_timeout=_POOL_TIMEOUT,
        connect_args={
            "connect_timeout": _CONNECT_TIMEOUT,
        },
        echo=False,
    )

    _engine_cache[connection_string] = engine
    return engine


def _dispose_engines():
    for engine in _engine_cache.values():
        try:
            engine.dispose()
        except Exception:
            pass
    _engine_cache.clear()


atexit.register(_dispose_engines)

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

        engine = _get_engine(db_user, db_pass, db_name)

        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        print(f"📅 Filtering from: {seven_days_ago}")

        with engine.connect() as conn:
            freshness = FreshnessTracker()
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
            time_column = _quote_identifier(columns["time"]) if columns.get("time") else None
            payment_column = (
                _quote_identifier(columns["payment_type"]) if columns.get("payment_type") else None
            )
            category_column = (
                _quote_identifier(columns["category"]) if columns.get("category") else None
            )

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
                return {"store": {"store_db": db_name, "store_id": selected_store.get("store_id")}, "sales": []}

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
            hourly_totals: Dict[int, float] = defaultdict(float)
            payment_totals: Dict[str, float] = defaultdict(float)
            category_totals: Dict[str, float] = defaultdict(float)
            total_sales_value = 0.0
            total_items_value = 0.0

            for row in items:
                date_value = row.get("sale_date")
                freshness.track(date_value)
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

                sale_time_value = row.get("sale_time")
                hour_from_time = _extract_hour(sale_time_value)
                if hour_from_time is None:
                    hour_from_time = _extract_hour(date_value)
                if hour_from_time is not None:
                    hourly_totals[hour_from_time] += price_value

                payment_label = row.get("payment_method")
                if payment_label is not None:
                    method_key = str(payment_label).strip() or "Unspecified"
                    payment_totals[method_key] += price_value

                category_label = row.get("category_label")
                if category_label is not None:
                    category_key = str(category_label).strip() or "Uncategorized"
                    category_totals[category_key] += price_value

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

            hourly_breakdown = [
                {
                    "hour": f"{hour:02d}:00",
                    "total_sales": round(amount, 2),
                }
                for hour, amount in sorted(hourly_totals.items())
            ]

            payment_breakdown = []
            if total_sales_value and payment_totals:
                payment_breakdown = [
                    {
                        "method": label,
                        "total_sales": round(amount, 2),
                        "percentage": round((amount / total_sales_value) * 100, 2),
                    }
                    for label, amount in sorted(payment_totals.items(), key=lambda item: item[1], reverse=True)
                ]

            category_breakdown = []
            if total_sales_value and category_totals:
                category_breakdown = [
                    {
                        "category": label,
                        "total_sales": round(amount, 2),
                        "percentage": round((amount / total_sales_value) * 100, 2),
                    }
                    for label, amount in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
                ]

            payload = {
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
                "breakdowns": {
                    "hourly": hourly_breakdown,
                    "payment_methods": payment_breakdown,
                    "categories": category_breakdown,
                },
                "purchase_orders": [],
                "inventory": None,
                "sales_history": [],
            }
            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info
            return payload

    except Exception as e:
        print(f"💥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")
