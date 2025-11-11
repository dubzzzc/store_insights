from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from collections import OrderedDict
import os
import atexit
import time
import logging

"""
This module re-exports the sales insights router that dynamically adapts to
Visual FoxPro schema differences (for example, DESCRIPTION vs DESCRIPT). The
main application continues to import the router from ``app.insights`` so we keep
that import path stable.
"""

router = APIRouter()

# Engine cache to reuse engines per database connection string
_engine_cache: "OrderedDict[str, Any]" = OrderedDict()


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, (int, float)):
        # treat as unix timestamp
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

    @property
    def latest(self) -> Optional[datetime]:
        return self._latest

    def to_payload(self) -> Optional[Dict[str, str]]:
        if not self._latest:
            return None
        iso = self._latest.isoformat()
        display = self._latest.strftime("%Y-%m-%d %H:%M:%S")
        return {"iso": iso, "display": display}


_MAX_CACHED_ENGINES = int(os.getenv("STORE_INSIGHTS_MAX_CACHED_ENGINES", "10"))
# Reduced pool size to prevent CPU overload on m5.large
_POOL_SIZE = int(os.getenv("STORE_INSIGHTS_POOL_SIZE", "10"))
_POOL_MAX_OVERFLOW = int(os.getenv("STORE_INSIGHTS_POOL_MAX_OVERFLOW", "10"))
_POOL_TIMEOUT = int(os.getenv("STORE_INSIGHTS_POOL_TIMEOUT", "30"))
_POOL_RECYCLE = int(os.getenv("STORE_INSIGHTS_POOL_RECYCLE", "1200"))
_CONNECT_TIMEOUT = int(os.getenv("STORE_INSIGHTS_CONNECT_TIMEOUT", "10"))
# Query execution timeout (seconds) - prevents runaway queries
_QUERY_TIMEOUT = int(os.getenv("STORE_INSIGHTS_QUERY_TIMEOUT", "30"))


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
        # Refresh LRU order so frequently used engines stay cached
        _engine_cache.move_to_end(connection_string)
        return engine

    # If we've hit the cache limit, dispose of the least-recently-used engine
    if _MAX_CACHED_ENGINES > 0 and len(_engine_cache) >= _MAX_CACHED_ENGINES:
        oldest_key, oldest_engine = _engine_cache.popitem(last=False)
        try:
            oldest_engine.dispose()
        except Exception:
            pass  # Ignore errors during disposal

    # Set query timeout via MySQL session variable to prevent runaway queries
    init_command = f"SET SESSION max_execution_time = {_QUERY_TIMEOUT * 1000}"  # MySQL uses milliseconds
    engine = create_engine(
        connection_string,
        pool_size=_POOL_SIZE,
        max_overflow=_POOL_MAX_OVERFLOW,
        pool_recycle=_POOL_RECYCLE,
        pool_pre_ping=True,
        pool_timeout=_POOL_TIMEOUT,
        connect_args={
            "connect_timeout": _CONNECT_TIMEOUT,
            "init_command": init_command,
        },
        echo=False,
    )

    _engine_cache[connection_string] = engine
    return engine


# Cleanup function to dispose of all engines on shutdown
def _dispose_engines():
    for engine in _engine_cache.values():
        try:
            engine.dispose()
        except Exception:
            pass
    _engine_cache.clear()


atexit.register(_dispose_engines)

# Configure logging for slow queries
_logger = logging.getLogger(__name__)
_SLOW_QUERY_THRESHOLD = float(
    os.getenv("STORE_INSIGHTS_SLOW_QUERY_THRESHOLD", "2.0")
)  # seconds


def _execute_with_timing(conn, query, params=None, query_name="query"):
    """
    Execute a query with timing and log slow queries for performance monitoring.
    """
    start_time = time.time()
    try:
        result = conn.execute(text(query), params or {})
        execution_time = time.time() - start_time

        # Log slow queries
        if execution_time > _SLOW_QUERY_THRESHOLD:
            _logger.warning(
                f"SLOW QUERY detected: {query_name} took {execution_time:.2f}s\n"
                f"Query: {query[:200]}...\n"
                f"Params: {params}"
            )

        return result
    except Exception as e:
        execution_time = time.time() - start_time
        _logger.error(
            f"QUERY ERROR after {execution_time:.2f}s: {query_name}\n"
            f"Query: {query[:200]}...\n"
            f"Error: {str(e)}"
        )
        raise


def _table_exists(conn, db_name: str, table: str) -> bool:
    res = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.tables
            WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
            LIMIT 1
            """
        ),
        {"db": db_name, "table": table},
    ).first()
    return bool(res)

def _get_columns(conn, db_name: str, table: str) -> List[str]:
    return [
        row[0].lower()
        for row in conn.execute(
            text(
                """
                SELECT COLUMN_NAME FROM information_schema.columns
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
                """
            ),
            {"db": db_name, "table": table},
        )
    ]

def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower = set(columns)
    for c in candidates:
        if c in lower:
            return c
    return None

def _normalize_payment_type(descript: str) -> str:
    """
    Normalize payment type from DESCRIPT field using comprehensive text matching.
    Handles many variations and abbreviations that stores might use.
    """
    if not descript:
        return "Other"
    
    desc_lower = str(descript).lower().strip()
    
    # Credit cards first (most specific)
    if any(x in desc_lower for x in ["american express", "amex", "am ex", "a/e", "a e", "amx", "amexcard"]):
        return "AmEx"
    if any(x in desc_lower for x in ["mastercard", "master card", "mc", "m/c", "m c", "mastr", "mstr", "mcard"]):
        return "MasterCard"
    if any(x in desc_lower for x in ["visa", "vsa", "vs"]):
        return "Visa"
    if any(x in desc_lower for x in ["discover", "dscvr", "dsc", "dis", "discover card", "discovercard"]):
        return "Discover"
    if any(x in desc_lower for x in ["debit", "db", "deb", "debit card", "debitcard", "debit crd"]):
        return "Debit"
    
    # Cash and Check
    if any(x in desc_lower for x in ["cash", "csh", "currency", "money"]):
        return "Cash"
    if any(x in desc_lower for x in ["check", "cheque", "chk", "chq", "ck", "chek"]):
        return "Check"
    
    # Gift cards
    if any(x in desc_lower for x in ["gift card", "giftcard", "gc", "gift cert", "giftcert", "gift certificate", "giftcertificate", "gift crd", "giftcrd"]):
        return "Gift Card"
    
    # House and EBT
    if any(x in desc_lower for x in ["house", "hse", "house account", "house acct", "house ac", "houseacct"]):
        return "House"
    if any(x in desc_lower for x in ["ebt", "electronic benefit", "food stamp", "foodstamp", "snap", "benefit", "ebt card", "ebtcard"]):
        return "EBT"
    
    return "Other"


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


def _detect_sales_source(conn, db_name: str) -> Optional[Dict[str, str]]:
    """
    Detect a sales-like table and appropriate columns for date, quantity, and amount.
    Returns a mapping with keys: table, date_col, qty_expr, amount_expr.
    qty_expr/amount_expr can be SQL expressions if columns are missing.
    """
    # Candidate table names in order of preference
    candidate_tables = [
        "sales",
        "sales_daily",
        "sales_history",
        "sales_journal",
        "jnl",
        "transactions",
        "pos_sales",
        "receipt",
    ]

    tables = set(
        row[0].lower()
        for row in conn.execute(
            text(
                """
                SELECT TABLE_NAME
                FROM information_schema.tables
                WHERE TABLE_SCHEMA = :db
                """
            ),
            {"db": db_name},
        )
    )

    table = next((t for t in candidate_tables if t in tables), None)
    if not table:
        return None

    # Fetch columns for the selected table
    column_names = set(
        row[0].lower()
        for row in conn.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM information_schema.columns
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
                """
            ),
            {"db": db_name, "table": table},
        )
    )

    # Pick best matching columns
    date_candidates = [
        "sale_date",
        "trans_date",
        "transaction_date",
        "tdate",
        "date",
        "created_at",
        "timestamp",
    ]
    qty_candidates = ["qty", "quantity", "items_sold", "total_qty", "count"]
    amount_candidates = [
        "amount",
        "total",
        "net_amount",
        "total_net",
        "total_sales",
        "net_total",
        "price",
    ]

    def pick(candidates):
        return next((c for c in candidates if c in column_names), None)

    date_col = pick(date_candidates)
    qty_col = pick(qty_candidates)
    amount_col = pick(amount_candidates)

    # Must have a date column to produce a time series
    if not date_col:
        return None

    # Build expressions with fallbacks if necessary
    # Prefer SUM(qty * pack) when both are present for item counts
    pack_col = "pack" if "pack" in column_names else None
    if qty_col and pack_col:
        qty_expr = f"SUM(`{qty_col}` * `{pack_col}`)"
    elif qty_col:
        qty_expr = f"SUM(`{qty_col}`)"
    else:
        qty_expr = "COUNT(*)"

    # Amount preference: direct amount/price, or compute qty*price
    price_col = "price" if "price" in column_names else None
    if amount_col:
        amount_expr = f"SUM(`{amount_col}`)"
    elif qty_col and price_col:
        amount_expr = f"SUM(`{qty_col}` * `{price_col}`)"
    else:
        amount_expr = "SUM(0)"

    return {
        "table": table,
        "date_col": date_col,
        "qty_expr": qty_expr,
        "amount_expr": amount_expr,
        "has_rflag": "rflag" in column_names,
        "has_sku": "sku" in column_names,
    }


@router.get("/sales")
def get_sales_insights(
    store: Optional[str] = Query(
        default=None, description="Store identifier (store_id or store_db)"
    ),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    # New: independent filters for purchase orders tab
    po_start: Optional[str] = Query(
        default=None, description="Purchase orders start date (YYYY-MM-DD)"
    ),
    po_end: Optional[str] = Query(
        default=None, description="Purchase orders end date (YYYY-MM-DD)"
    ),
    user: dict = Depends(get_auth_user),
):
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        # Get engine with connection pooling
        engine = _get_engine(db_user, db_pass, db_name)

        with engine.connect() as conn:
            freshness = FreshnessTracker()
            detected = _detect_sales_source(conn, db_name)

            if not detected:
                # No suitable table/columns found: return empty but valid payload
                sales_data = []
                response_meta = {
                    "store_db": db_name,
                    "store_id": selected_store.get("store_id"),
                    "store_name": selected_store.get("store_name"),
                }
                return {"store": response_meta, "sales": sales_data}

            table = detected["table"]
            date_col = detected["date_col"]
            qty_expr = detected["qty_expr"]
            amount_expr = detected["amount_expr"]

            # Get column names to determine which columns to select in subquery
            table_cols = _get_columns(conn, db_name, table)
            # Select columns that might be referenced in expressions (qty, pack, price, amount, date)
            select_cols = [f"`{date_col}`"]
            for col in ["qty", "pack", "price", "amount", "total"]:
                if col in table_cols:
                    select_cols.append(f"`{col}`")
            select_cols_str = ", ".join(select_cols)

            # Optional filters to exclude non-item or return rows if present
            where_clauses = []
            if detected.get("has_rflag"):
                where_clauses.append("`rflag` = 0")
            if detected.get("has_sku"):
                where_clauses.append("`sku` > 0")
            # Date range filters using half-open interval for index usage
            params = {}
            if start:
                where_clauses.append(f"`{date_col}` >= :start_dt")
                params["start_dt"] = f"{start} 00:00:00"
            if end:
                try:
                    from datetime import timedelta

                    end_next = (
                        datetime.fromisoformat(end) + timedelta(days=1)
                    ).strftime("%Y-%m-%d 00:00:00")
                except Exception:
                    end_next = f"{end} 23:59:59"
                where_clauses.append(f"`{date_col}` < :end_dt")
                params["end_dt"] = end_next
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            # Optimize: Aggregate directly without subquery to avoid materializing large intermediate results
            # WHERE clause uses index-friendly half-open intervals, DATE() only in SELECT/GROUP BY
            sql = f"""
                SELECT 
                    DATE(`{date_col}`) AS date,
                    {qty_expr} AS total_items_sold,
                    {amount_expr} AS total_sales
                FROM `{table}`
                {where_sql}
                GROUP BY DATE(`{date_col}`)
                ORDER BY DATE(`{date_col}`) DESC
                {"LIMIT 7" if not (start or end) else ""}
            """

            result = conn.execute(text(sql), params).mappings()

            sales_data = []
            for row in result:
                freshness.track(row.get("date"))
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

            # Get tax total from LINE 941 (before tenders) for the selected date range
            tax_total = 0.0
            if _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or None
                jnl_amt_col = None
                for c in ["amount", "total", "price"]:
                    if c in jnl_cols:
                        jnl_amt_col = c
                        break
                jnl_date_filter_col = (
                    _pick_column(
                        jnl_cols,
                        [
                            "sale_date",
                            "trans_date",
                            "transaction_date",
                            "tdate",
                            "date",
                        ],
                    )
                    or None
                )
                jnl_rflag = "rflag" if "rflag" in jnl_cols else None

                if jnl_line_col and jnl_amt_col:
                    tax_where = []
                    tax_params = {}
                    if jnl_date_filter_col and start:
                        tax_where.append(
                            f"jnl.`{jnl_date_filter_col}` >= :tax_start_dt"
                        )
                        tax_params["tax_start_dt"] = f"{start} 00:00:00"
                    if jnl_date_filter_col and end:
                        try:
                            from datetime import timedelta

                            tax_end_next = (
                                datetime.fromisoformat(end) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            tax_end_next = f"{end} 23:59:59"
                        tax_where.append(f"jnl.`{jnl_date_filter_col}` < :tax_end_dt")
                        tax_params["tax_end_dt"] = tax_end_next
                    if jnl_rflag:
                        tax_where.append(f"jnl.`{jnl_rflag}` <= 0")
                    tax_where.append(f"jnl.`{jnl_line_col}` = 941")

                    tax_sql = f"""
                        SELECT SUM(jnl.`{jnl_amt_col}`) AS total
                        FROM jnl
                        WHERE {' AND '.join(tax_where) if tax_where else '1=1'}
                    """
                    tax_row = conn.execute(text(tax_sql), tax_params).mappings().first()
                    if tax_row:
                        tax_total = float(tax_row.get("total") or 0)

            # Build summary
            gross_total = sum(row["total_sales"] for row in sales_data)
            total_items = sum(row["total_items_sold"] for row in sales_data)
            days_captured = len(sales_data)

            # Count distinct sales (transactions) for average dollars per sale calculation
            sale_count = 0
            if _table_exists(conn, db_name, "jnh"):
                jnh_cols = _get_columns(conn, db_name, "jnh")
                jnh_time = (
                    _pick_column(jnh_cols, ["tstamp", "timestamp", "time", "t_time"])
                    or "tstamp"
                )
                jnh_sale = (
                    _pick_column(jnh_cols, ["sale", "sale_id", "invno"]) or "sale"
                )
                jnh_total = _pick_column(jnh_cols, ["total", "amount", "price"]) or None

                if jnh_total:
                    sale_count_where = []
                    sale_count_params: Dict[str, Any] = {}

                    # Filter by date using jnh.tstamp (half-open interval)
                    if start:
                        sale_count_where.append(f"jnh.`{jnh_time}` >= :sc_start_dt")
                        sale_count_params["sc_start_dt"] = f"{start} 00:00:00"
                    if end:
                        try:
                            from datetime import timedelta

                            sc_end_next = (
                                datetime.fromisoformat(end) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            sc_end_next = f"{end} 23:59:59"
                        sale_count_where.append(f"jnh.`{jnh_time}` < :sc_end_dt")
                        sale_count_params["sc_end_dt"] = sc_end_next

                    # Count distinct sales, excluding void sales
                    if _table_exists(conn, db_name, "jnl"):
                        jnl_cols = _get_columns(conn, db_name, "jnl")
                        jnl_sale_col = (
                            _pick_column(jnl_cols, ["sale", "sale_id", "invno"])
                            or "sale"
                        )
                        jnl_rflag = "rflag" if "rflag" in jnl_cols else None
                        jnl_sku_col = (
                            _pick_column(jnl_cols, ["sku", "item", "item_id"]) or "sku"
                        )

                        void_sales_where = []
                        if jnl_rflag:
                            void_sales_where.append(f"jnl_void.`{jnl_rflag}` = 4")
                        void_sales_where.append(f"jnl_void.`{jnl_sku_col}` > 0")

                        sale_count_sql = f"""
                            SELECT COUNT(DISTINCT jnh.`{jnh_sale}`) AS sale_count
                            FROM jnh
                            WHERE {' AND '.join(sale_count_where) if sale_count_where else '1=1'}
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM jnl jnl_void
                                  WHERE jnl_void.`{jnl_sale_col}` = jnh.`{jnh_sale}`
                                    AND {' AND '.join(void_sales_where)}
                                  LIMIT 1
                              )
                        """
                        sale_count_row = (
                            conn.execute(text(sale_count_sql), sale_count_params)
                            .mappings()
                            .first()
                        )
                        if sale_count_row:
                            sale_count = int(sale_count_row.get("sale_count") or 0)

            # Initialize total_net - will be updated after payment_methods are calculated
            total_net = 0.0

            summary = {
                "gross_sales": round(gross_total, 2),
                "total_items": float(total_items),
                "days_captured": days_captured,
                "average_dollars_per_sale": (
                    round(gross_total / sale_count, 2) if sale_count > 0 else 0.0
                ),
                "tax_total": round(tax_total, 2),
                "total_net": 0.0,  # Will be updated after payment_methods calculation
            }

            # Hourly breakdown: use jnl tender lines (LINE 980-989) with RFLAG <= 0, filter by jnl.DATE,
            # join to jnh to get hour from jnh.tstamp, sum jnl.PRICE. Matches process_prefix logic.
            hourly: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl") and _table_exists(
                conn, db_name, "jnh"
            ):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_sale = (
                    _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or "sale"
                )
                jnl_date_col = (
                    _pick_column(
                        jnl_cols,
                        [
                            "sale_date",
                            "trans_date",
                            "transaction_date",
                            "tdate",
                            "date",
                        ],
                    )
                    or None
                )
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_price_col = (
                    _pick_column(jnl_cols, ["price", "amount", "total"]) or None
                )
                jnl_rflag = "rflag" if "rflag" in jnl_cols else None

                jnh_cols = _get_columns(conn, db_name, "jnh")
                jnh_time = (
                    _pick_column(jnh_cols, ["tstamp", "timestamp", "time", "t_time"])
                    or "tstamp"
                )
                jnh_sale = (
                    _pick_column(jnh_cols, ["sale", "sale_id", "invno"]) or jnl_sale
                )
                jnh_total = _pick_column(jnh_cols, ["total", "amount", "price"]) or None

                if jnl_date_col and jnl_line_col and jnh_total:
                    where_parts = []
                    params2: Dict[str, Any] = {}

                    # Filter by date using jnh.tstamp (half-open interval)
                    if start:
                        where_parts.append(f"jnh.`{jnh_time}` >= :h_start_dt")
                        params2["h_start_dt"] = f"{start} 00:00:00"
                    if end:
                        try:
                            from datetime import timedelta

                            h_end_next = (
                                datetime.fromisoformat(end) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            h_end_next = f"{end} 23:59:59"
                        where_parts.append(f"jnh.`{jnh_time}` < :h_end_dt")
                        params2["h_end_dt"] = h_end_next

                    # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                    void_sales_where = []
                    if jnl_rflag:
                        void_sales_where.append(f"jnl_void.`{jnl_rflag}` = 4")
                    jnl_sku_col = (
                        _pick_column(jnl_cols, ["sku", "item", "item_id"]) or "sku"
                    )
                    void_sales_where.append(f"jnl_void.`{jnl_sku_col}` > 0")

                    # Sales must have at least one tender line (980-989) with rflag <= 0
                    tender_check_where = [
                        f"jnl_tender.`{jnl_line_col}` BETWEEN 980 AND 989"
                    ]
                    if jnl_rflag:
                        tender_check_where.append(f"jnl_tender.`{jnl_rflag}` <= 0")

                    # Use EXISTS subqueries - more efficient than JOINs for this use case
                    hourly_sql = f"""
                        SELECT HOUR(jnh.`{jnh_time}`) AS hour, SUM(jnh.`{jnh_total}`) AS total_sales
                        FROM jnh
                        WHERE {' AND '.join(where_parts)}
                          AND EXISTS (
                              SELECT 1
                              FROM jnl jnl_tender
                              WHERE jnl_tender.`{jnl_sale}` = jnh.`{jnh_sale}`
                                AND {' AND '.join(tender_check_where)}
                              LIMIT 1
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM jnl jnl_void
                              WHERE jnl_void.`{jnl_sale}` = jnh.`{jnh_sale}`
                                AND {' AND '.join(void_sales_where)}
                              LIMIT 1
                          )
                        GROUP BY HOUR(jnh.`{jnh_time}`)
                        ORDER BY hour
                    """
                    rows = conn.execute(text(hourly_sql), params2).mappings()
                    hourly = []
                    for r in rows:
                        hour_num = int(r["hour"])
                        if hour_num == 12:
                            hour_label = "NOON TO 12:59 PM"
                        elif hour_num == 0:
                            hour_label = "12:00 AM TO 12:59 AM"
                        elif hour_num < 12:
                            hour_label = f"{hour_num}:00 TO {hour_num}:59 AM"
                        else:
                            # PM hours: 13->1, 14->2, etc.
                            pm_hour = hour_num - 12
                            hour_label = f"{pm_hour}:00 TO {pm_hour}:59 PM"
                        hourly.append(
                            {
                                "hour": hour_label,
                                "total_sales": float(r["total_sales"]),
                            }
                        )

            # Payment methods from jnl tenders LINE 980-989
            # Filter dates using jnh.tstamp to match hourly sales logic for consistency
            # Shows ALL payment types/tenders for the selected date range, including gift cards
            # LINE 980-989 are just tender slots (order received), payment type is identified by CAT column
            # For cash payments, subtract CASH CHANGE (LINE 999) from the total
            payment_methods: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl") and _table_exists(
                conn, db_name, "jnh"
            ):
                cols = _get_columns(conn, db_name, "jnl")
                line_col = _pick_column(cols, ["line", "line_code"]) or None
                cat_col = "cat" if "cat" in cols else None
                descript_col = (
                    _pick_column(cols, ["descript", "description", "desc"]) or None
                )
                jnl_rflag = "rflag" if "rflag" in cols else None
                amt_col = None
                for c in ["amount", "total", "price"]:
                    if c in cols:
                        amt_col = c
                        break

                # Get jnh columns for date filtering
                jnh_cols = _get_columns(conn, db_name, "jnh")
                jnh_time = (
                    _pick_column(jnh_cols, ["tstamp", "timestamp", "time", "t_time"])
                    or "tstamp"
                )
                jnh_sale = (
                    _pick_column(jnh_cols, ["sale", "sale_id", "invno"]) or "sale"
                )
                jnl_sale_col = (
                    _pick_column(cols, ["sale", "sale_id", "invno"]) or "sale"
                )

                where_parts = []
                params3: Dict[str, Any] = {}
                # Date filter: use jnh.tstamp to match hourly sales logic
                if start:
                    where_parts.append(f"jnh.`{jnh_time}` >= :p_start_dt")
                    params3["p_start_dt"] = f"{start} 00:00:00"
                if end:
                    try:
                        from datetime import timedelta

                        p_end_next = (
                            datetime.fromisoformat(end) + timedelta(days=1)
                        ).strftime("%Y-%m-%d 00:00:00")
                    except Exception:
                        p_end_next = f"{end} 23:59:59"
                    where_parts.append(f"jnh.`{jnh_time}` < :p_end_dt")
                    params3["p_end_dt"] = p_end_next
                # Filter: RFLAG <= 0 (excludes returns/voids, includes all valid tenders)
                if jnl_rflag:
                    where_parts.append(f"jnl.`{jnl_rflag}` <= 0")

                if line_col and amt_col:
                    # Get sale identifier column to match CASH CHANGE to sales with cash tenders
                    jnl_sale_col = (
                        _pick_column(cols, ["sale", "sale_id", "invno"]) or "sale"
                    )
                    # Get SKU column for void sales check
                    jnl_sku_col = (
                        _pick_column(cols, ["sku", "item", "item_id"]) or "sku"
                    )

                    # Get ALL tenders (LINE 980-989) - payment type is identified by jnl.cat
                    # For ecommerce sales, cat may be in promo column, so use promo as fallback
                    # Join to cat table to get cat.name for payment method names
                    # Ecommerce sales have cat.name containing "WEB" - include these in payment methods
                    promo_col = "promo" if "promo" in cols else None
                    if cat_col:
                        # Check if cat table exists and get column names
                        cat_table_exists = _table_exists(conn, db_name, "cat")
                        if cat_table_exists:
                            cat_table_cols = _get_columns(conn, db_name, "cat")
                            cat_code_col = (
                                _pick_column(cat_table_cols, ["cat", "code", "id"])
                                or "cat"
                            )
                            cat_name_col = (
                                _pick_column(
                                    cat_table_cols,
                                    ["name", "desc", "description", "label"],
                                )
                                or "name"
                            )

                            # Join jnl to cat table: jnl.cat = cat.cat to get cat.name
                            # Use cat.name, fallback to descript, then cat code
                            # For cash: subtract CASH CHANGE (LINE 999) per sale using correlated subquery
                            descript_fallback = (
                                f"jnl.`{descript_col}`" if descript_col else "NULL"
                            )
                            descript_group = (
                                f", jnl.`{descript_col}`" if descript_col else ""
                            )

                            # Build cash identification condition
                            cash_check_where = []
                            if descript_col:
                                cash_check_where.append(
                                    f"UPPER(jnl_cash.`{descript_col}`) LIKE '%CASH%'"
                                )
                            cash_check_where.append(
                                f"UPPER(cat_cash.`{cat_name_col}`) LIKE '%CASH%'"
                            )
                            cash_check_condition = " OR ".join(cash_check_where)

                            # Build cat lookup: use cat column, fallback to promo if cat is empty/0
                            # This handles ecommerce sales where cat might be in promo column
                            cat_lookup_expr = (
                                f"COALESCE(NULLIF(jnl.`{cat_col}`, 0), NULLIF(jnl.`{promo_col}`, 0), 0)"
                                if promo_col
                                else f"COALESCE(NULLIF(jnl.`{cat_col}`, 0), 0)"
                            )

                            # Identify if current tender is cash (using subquery to avoid JOIN duplicates)
                            cash_identify_condition = f"UPPER(COALESCE((SELECT cat.`{cat_name_col}` FROM cat WHERE cat.`{cat_code_col}` = {cat_lookup_expr} LIMIT 1), '')) LIKE '%CASH%'"
                            if descript_col:
                                cash_identify_condition = f"({cash_identify_condition} OR UPPER(jnl.`{descript_col}`) LIKE '%CASH%')"

                            # Optimize: Pre-calculate cash change per sale using a simple subquery JOIN
                            # Get cash change (LINE 999) only for sales that have cash tenders and are not void
                            # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                            # Use jnh.tstamp for date filtering to match main query
                            cash_change_where = [
                                f"jnl_cc.`{line_col}` = 999",
                                f"jnl_cc.`{jnl_rflag}` <= 0",
                            ]
                            if start:
                                cash_change_where.append(
                                    f"jnh_cc.`{jnh_time}` >= :cc_start_dt"
                                )
                            if end:
                                cash_change_where.append(
                                    f"jnh_cc.`{jnh_time}` < :cc_end_dt"
                                )

                            # Build void sales check for cash change subquery
                            cash_change_void_where = []
                            if jnl_rflag:
                                cash_change_void_where.append(
                                    f"jnl_void_cc.`{jnl_rflag}` = 4"
                                )
                            cash_change_void_where.append(
                                f"jnl_void_cc.`{jnl_sku_col}` > 0"
                            )

                            # Build cash tender check - ensure sale has cash tenders
                            cash_tender_check_where = [
                                f"jnl_cash_tender.`{line_col}` BETWEEN 980 AND 989"
                            ]
                            if jnl_rflag:
                                cash_tender_check_where.append(
                                    f"jnl_cash_tender.`{jnl_rflag}` <= 0"
                                )

                            # Build cash identification condition for EXISTS check
                            cash_tender_identify = [
                                f"UPPER(COALESCE(cat_cash_tender.`{cat_name_col}`, '')) LIKE '%CASH%'"
                            ]
                            if descript_col:
                                cash_tender_identify.append(
                                    f"UPPER(jnl_cash_tender.`{descript_col}`) LIKE '%CASH%'"
                                )
                            cash_tender_identify_condition = " OR ".join(
                                cash_tender_identify
                            )

                            # Use subquery for cat.name to avoid duplicates from multiple cat rows
                            # This ensures we only get one cat.name per cat code
                            # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                            void_sales_where = []
                            if jnl_rflag:
                                void_sales_where.append(f"jnl_void.`{jnl_rflag}` = 4")
                            void_sales_where.append(f"jnl_void.`{jnl_sku_col}` > 0")

                            tender_sql = f"""
                                SELECT 
                                    {cat_lookup_expr} AS code,
                                    COALESCE(
                                        (SELECT cat.`{cat_name_col}` FROM cat WHERE cat.`{cat_code_col}` = {cat_lookup_expr} LIMIT 1),
                                        {descript_fallback},
                                        CAST({cat_lookup_expr} AS CHAR)
                                    ) AS method_name,
                                    SUM(
                                        CASE 
                                            WHEN (
                                                UPPER(COALESCE((SELECT cat.`{cat_name_col}` FROM cat WHERE cat.`{cat_code_col}` = {cat_lookup_expr} LIMIT 1), '')) LIKE '%CASH%'
                                                OR UPPER(jnl.`{descript_col}`) LIKE '%CASH%'
                                            ) THEN 
                                                GREATEST(0, jnl.`{amt_col}` - ABS(COALESCE(cash_change_per_sale.cash_change, 0)))
                                            ELSE 
                                                jnl.`{amt_col}`
                                        END
                                    ) AS total
                                FROM jnl
                                INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                                LEFT JOIN (
                                    SELECT 
                                        jnl_cc.`{jnl_sale_col}` AS sale_id,
                                        SUM(jnl_cc.`{amt_col}`) AS cash_change
                                    FROM jnl jnl_cc
                                    INNER JOIN jnh jnh_cc ON jnh_cc.`{jnh_sale}` = jnl_cc.`{jnl_sale_col}`
                                    WHERE {' AND '.join(cash_change_where)}
                                      AND NOT EXISTS (
                                          SELECT 1
                                          FROM jnl jnl_void_cc
                                          WHERE jnl_void_cc.`{jnl_sale_col}` = jnl_cc.`{jnl_sale_col}`
                                            AND {' AND '.join(cash_change_void_where)}
                                          LIMIT 1
                                      )
                                      AND EXISTS (
                                          SELECT 1
                                          FROM jnl jnl_cash_tender
                                          LEFT JOIN cat cat_cash_tender ON cat_cash_tender.`{cat_code_col}` = jnl_cash_tender.`{cat_col}`
                                          WHERE jnl_cash_tender.`{jnl_sale_col}` = jnl_cc.`{jnl_sale_col}`
                                            AND {' AND '.join(cash_tender_check_where)}
                                            AND ({cash_tender_identify_condition})
                                          LIMIT 1
                                      )
                                    GROUP BY jnl_cc.`{jnl_sale_col}`
                                ) AS cash_change_per_sale ON cash_change_per_sale.sale_id = jnl.`{jnl_sale_col}`
                                WHERE {' AND '.join(where_parts) if where_parts else '1=1'} 
                                  AND jnl.`{line_col}` BETWEEN 980 AND 989
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM jnl jnl_void
                                      WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                        AND {' AND '.join(void_sales_where)}
                                      LIMIT 1
                                  )
                                GROUP BY {cat_lookup_expr}
                                ORDER BY {cat_lookup_expr}
                            """

                            # Add parameters for cash change subquery (use same as main query)
                            if start:
                                params3["cc_start_dt"] = params3["p_start_dt"]
                            if end:
                                params3["cc_end_dt"] = params3["p_end_dt"]
                        else:
                            # No cat table: use DESCRIPT as fallback
                            # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                            void_sales_where_fallback = []
                            if jnl_rflag:
                                void_sales_where_fallback.append(
                                    f"jnl_void.`{jnl_rflag}` = 4"
                                )
                            void_sales_where_fallback.append(
                                f"jnl_void.`{jnl_sku_col}` > 0"
                            )

                            if descript_col:
                                tender_sql = f"""
                                    SELECT 
                                        jnl.`{cat_col}` AS code,
                                        jnl.`{descript_col}` AS method_name,
                                        SUM(jnl.`{amt_col}`) AS total
                                    FROM jnl
                                    INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                                    WHERE {' AND '.join(where_parts) if where_parts else '1=1'} 
                                      AND jnl.`{line_col}` BETWEEN 980 AND 989
                                      AND NOT EXISTS (
                                          SELECT 1
                                          FROM jnl jnl_void
                                          WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                            AND {' AND '.join(void_sales_where_fallback)}
                                          LIMIT 1
                                      )
                                    GROUP BY jnl.`{cat_col}`, jnl.`{descript_col}`
                                    ORDER BY jnl.`{cat_col}`
                                """
                            else:
                                # Last resort: just use CAT code
                                tender_sql = f"""
                                    SELECT 
                                        jnl.`{cat_col}` AS code,
                                        CAST(jnl.`{cat_col}` AS CHAR) AS method_name,
                                        SUM(jnl.`{amt_col}`) AS total
                                    FROM jnl
                                    INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                                    WHERE {' AND '.join(where_parts) if where_parts else '1=1'} 
                                      AND jnl.`{line_col}` BETWEEN 980 AND 989
                                      AND NOT EXISTS (
                                          SELECT 1
                                          FROM jnl jnl_void
                                          WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                            AND {' AND '.join(void_sales_where_fallback)}
                                          LIMIT 1
                                      )
                                    GROUP BY jnl.`{cat_col}`
                                    ORDER BY jnl.`{cat_col}`
                                """
                    else:
                        # No CAT column: use DESCRIPT
                        # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                        void_sales_where_no_cat = []
                        if jnl_rflag:
                            void_sales_where_no_cat.append(
                                f"jnl_void.`{jnl_rflag}` = 4"
                            )
                        void_sales_where_no_cat.append(f"jnl_void.`{jnl_sku_col}` > 0")

                        if descript_col:
                            tender_sql = f"""
                                SELECT 
                                    jnl.`{descript_col}` AS code,
                                    jnl.`{descript_col}` AS method_name,
                                    SUM(jnl.`{amt_col}`) AS total
                                FROM jnl
                                INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                                WHERE {' AND '.join(where_parts) if where_parts else '1=1'} 
                                  AND jnl.`{line_col}` BETWEEN 980 AND 989
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM jnl jnl_void
                                      WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                        AND {' AND '.join(void_sales_where_no_cat)}
                                      LIMIT 1
                                  )
                                GROUP BY jnl.`{descript_col}`
                                ORDER BY jnl.`{descript_col}`
                            """
                        else:
                            # Last resort: group by LINE (not ideal)
                            tender_sql = f"""
                                SELECT 
                                    jnl.`{line_col}` AS code,
                                    CAST(jnl.`{line_col}` AS CHAR) AS method_name,
                                    SUM(jnl.`{amt_col}`) AS total
                                FROM jnl
                                INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                                WHERE {' AND '.join(where_parts) if where_parts else '1=1'} 
                                  AND jnl.`{line_col}` BETWEEN 980 AND 989
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM jnl jnl_void
                                      WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                        AND {' AND '.join(void_sales_where_no_cat)}
                                      LIMIT 1
                                  )
                                GROUP BY jnl.`{line_col}`
                                ORDER BY jnl.`{line_col}`
                            """
                    rows = conn.execute(text(tender_sql), params3).mappings()

                    # Process payment methods - cash change is already handled in SQL query
                    # All payment types (including cash) use the same calculation logic
                    total_sum = 0.0
                    tmp = []
                    for r in rows:
                        code_raw = r.get("code")
                        method_name = r.get("method_name") or str(code_raw) or "Unknown"
                        amount = float(r.get("total") or 0)
                        # Amount is already correct (cash change subtracted in SQL for cash, others unchanged)
                        total_sum += amount
                        # Use method_name from query (from cat.name or descript)
                        tmp.append({"method": str(method_name), "total_sales": amount})
                    # Calculate percentages based on total_sum (which includes adjusted cash, not raw cash)
                    if total_sum:
                        payment_methods = [
                            {**item, "percentage": round((item["total_sales"]/total_sum)*100, 2)} for item in tmp
                        ]
                        # Update total_net in summary - sum of all payment methods (tenders)
                        total_net = total_sum
                        summary["total_net"] = round(total_net, 2)

            # Categories: aggregate by jnl.cat for ALL product sales (sku > 0) in the selected date range
            # Shows ALL categories with sales for the date range, joined to cat table for category names
            categories: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl"):
                cols = _get_columns(conn, db_name, "jnl")
                if "cat" in cols:
                    # Use same date column and filters as payment methods
                    jnl_date_col = _pick_column(cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                    jnl_rflag = "rflag" if "rflag" in cols else None
                    jnl_sku_col = (
                        _pick_column(cols, ["sku", "item", "item_id"]) or "sku"
                    )
                    amt_col = None
                    for c in ["amount", "total", "price"]:
                        if c in cols:
                            amt_col = c
                            break

                    where_parts = []
                    params4: Dict[str, Any] = {}
                    # Date filter: show ALL product sales for the selected date range
                    if jnl_date_col and start:
                        where_parts.append(f"jnl.`{jnl_date_col}` >= :c_start_dt")
                        params4["c_start_dt"] = f"{start} 00:00:00"
                    if jnl_date_col and end:
                        try:
                            from datetime import timedelta

                            c_end_next = (
                                datetime.fromisoformat(end) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            c_end_next = f"{end} 23:59:59"
                        where_parts.append(f"jnl.`{jnl_date_col}` < :c_end_dt")
                        params4["c_end_dt"] = c_end_next
                    # Filter: RFLAG <= 0 (excludes returns/voids, includes all valid product sales)
                    if jnl_rflag:
                        where_parts.append(f"jnl.`{jnl_rflag}` <= 0")
                    # Only product sales (sku > 0) - excludes tender lines, includes all actual products
                    where_parts.append(f"jnl.`{jnl_sku_col}` > 0")

                    if amt_col:
                        # Join to cat table to get category name (cat.name)
                        cat_join_label = None
                        if _table_exists(conn, db_name, "cat"):
                            cat_cols = _get_columns(conn, db_name, "cat")
                            cat_code = (
                                _pick_column(cat_cols, ["cat", "code", "id"]) or "cat"
                            )
                            # Prefer "name" as specified by user, fallback to desc/description
                            cat_label = (
                                _pick_column(
                                    cat_cols, ["name", "desc", "description", "label"]
                                )
                                or cat_code
                            )
                            cat_join_label = (cat_code, cat_label)

                        if cat_join_label:
                            cc, cl = cat_join_label
                            # Use subquery for cat.name to avoid duplicates from multiple cat rows
                            cat_sql = f"""
                                SELECT COALESCE(
                                    (SELECT cat.`{cl}` FROM cat WHERE cat.`{cc}` = jnl.`cat` LIMIT 1),
                                    CAST(jnl.`cat` AS CHAR)
                                ) AS category, 
                                SUM(jnl.`{amt_col}`) AS total
                                FROM jnl
                                WHERE {' AND '.join(where_parts)}
                                GROUP BY jnl.`cat`
                                ORDER BY total DESC
                                LIMIT 50
                            """
                        else:
                            cat_sql = f"""
                                SELECT CAST(jnl.`cat` AS CHAR) AS category, 
                                       SUM(jnl.`{amt_col}`) AS total
                                FROM jnl
                                WHERE {' AND '.join(where_parts)}
                                GROUP BY jnl.`cat`
                                ORDER BY total DESC
                                LIMIT 50
                            """
                        rows = conn.execute(text(cat_sql), params4).mappings()
                        categories = [
                            {
                                "category": str(r["category"]),
                                "total_sales": float(r["total"]),
                            }
                            for r in rows
                        ]

            # Top products from hst (explicit column usage with safe fallbacks)
            top_items: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "hst"):
                hst_cols = _get_columns(conn, db_name, "hst")
                hst_date = _pick_column(hst_cols, ["date", "tdate", "sale_date", "trans_date"]) or "date"
                hst_tstamp = (
                    _pick_column(hst_cols, ["tstamp", "timestamp", "time", "t_time"])
                    or None
                )
                hst_qty = _pick_column(hst_cols, ["qty", "quantity"]) or "qty"
                hst_price = (
                    _pick_column(hst_cols, ["price", "amount", "total"]) or "price"
                )
                hst_cost = _pick_column(hst_cols, ["cost", "lcost", "acost"]) or None
                hst_pack = _pick_column(hst_cols, ["pack", "mult", "casepack"]) or None
                hst_sku = _pick_column(hst_cols, ["sku"]) or "sku"

                where_parts = [f"`{hst_sku}` > 0"]
                params5: Dict[str, Any] = {}
                if start:
                    where_parts.append(f"`{hst_date}` >= :t_start_dt")
                    params5["t_start_dt"] = f"{start} 00:00:00"
                if end:
                    try:
                        from datetime import timedelta

                        t_end_next = (
                            datetime.fromisoformat(end) + timedelta(days=1)
                        ).strftime("%Y-%m-%d 00:00:00")
                    except Exception:
                        t_end_next = f"{end} 23:59:59"
                    where_parts.append(f"`{hst_date}` < :t_end_dt")
                    params5["t_end_dt"] = t_end_next

                # Build expressions (items sold = packs / pack_size when pack exists)
                if hst_pack in hst_cols:
                    qty_expr = f"SUM(COALESCE(`{hst_qty}`,0) / NULLIF(COALESCE(`{hst_pack}`,1),0))"
                else:
                    qty_expr = f"SUM(COALESCE(`{hst_qty}`,0))"
                sales_expr = f"SUM(COALESCE(`{hst_price}`,0))"
                cost_expr = (
                    f"SUM(COALESCE(`{hst_cost}`,0) * COALESCE(`{hst_qty}`,0))"
                    if hst_cost in hst_cols
                    else "SUM(0)"
                )

                base_sql = f"""
                    SELECT `{hst_sku}` AS sku,
                           {qty_expr} AS total_qty,
                           {sales_expr} AS total_sales,
                           {cost_expr} AS total_cost
                    FROM hst
                    WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                    GROUP BY `{hst_sku}`
                    ORDER BY total_qty DESC, total_sales DESC
                    LIMIT 20
                """
                list_rows = list(conn.execute(text(base_sql), params5).mappings())

                # Join descriptions from inv
                inv_names: Dict[str, str] = {}
                skus = [str(r["sku"]) for r in list_rows]
                if skus and _table_exists(conn, db_name, "inv"):
                    inv_cols = _get_columns(conn, db_name, "inv")
                    inv_sku = _pick_column(inv_cols, ["sku"]) or "sku"
                    inv_name = _pick_column(inv_cols, ["desc", "description", "name"]) or inv_sku
                    in_params = {"skus": tuple(skus)}
                    inv_sql = text(f"SELECT `{inv_sku}` AS sku, `{inv_name}` AS name FROM inv WHERE `{inv_sku}` IN :skus").bindparams()
                    inv_rows = conn.execute(inv_sql, in_params).mappings()
                    for r in inv_rows:
                        inv_names[str(r["sku"])] = (
                            str(r["name"]) if r["name"] is not None else ""
                        )

                # Map results
                top_items = [
                    {
                        "sku": str(r["sku"]),
                        "description": inv_names.get(str(r["sku"]), ""),
                        "total_items_sold": float(r.get("total_qty") or 0.0),
                        "total_sales": float(r.get("total_sales") or 0.0),
                        "total_cost": float(r.get("total_cost") or 0.0),
                    }
                    for r in list_rows
                ]

            # Purchase orders: POs grouped by vendor (independent date filter)
            # poh.vendor matches vnd.vendor, use vnd.lastname as vendor name
            purchase_orders: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "poh"):
                poh_cols = _get_columns(conn, db_name, "poh")
                poh_status_col = _pick_column(poh_cols, ["status", "stat"]) or "status"
                poh_vendor_col = _pick_column(poh_cols, ["vendor", "vcode"]) or "vendor"
                poh_total_col = _pick_column(poh_cols, ["total"]) or None
                poh_rcv_col = _pick_column(poh_cols, ["rcvdate", "received_date", "rcv_date"]) or None
                poh_ord_col = _pick_column(poh_cols, ["orddate", "order_date"]) or None
                poh_id_col = _pick_column(poh_cols, ["id", "po_id", "poh_id", "invno"]) or None

                # Independent date filter for POs - ONLY use po_start/po_end, never fall back to start/end
                # This prevents cross-contamination between sales date selector and PO date selector
                po_start_eff = po_start
                po_end_eff = po_end

                # Group by vendor using provided PO date range (all statuses: 3=Complete, 4=posted, 6=received, 8=open order)
                # Optimized query structure to use composite index (status, rcvdate, vendor)
                where_parts = [f"poh.`{poh_status_col}` IN (3, 4, 6, 8)"]
                params: Dict[str, Any] = {}

                # Date filters: use rcvdate only (posted/received on date range)
                # Use AND (not OR) to match index structure and ensure proper date range filtering
                if poh_rcv_col and (po_start_eff or po_end_eff):
                    if po_start_eff:
                        where_parts.append(f"poh.`{poh_rcv_col}` >= :po_start_dt")
                        params["po_start_dt"] = f"{po_start_eff} 00:00:00"
                    if po_end_eff:
                        try:
                            from datetime import timedelta

                            po_end_next = (
                                datetime.fromisoformat(po_end_eff) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            po_end_next = f"{po_end_eff} 23:59:59"
                        where_parts.append(f"poh.`{poh_rcv_col}` < :po_end_dt")
                        params["po_end_dt"] = po_end_next

                # Join vnd: poh.vendor = vnd.vendor, use vnd.lastname
                vnd_cols = _get_columns(conn, db_name, "vnd") if _table_exists(conn, db_name, "vnd") else []
                vnd_vendor_col = _pick_column(vnd_cols, ["vendor"])
                vnd_lastname_col = _pick_column(vnd_cols, ["lastname", "last_name", "lname"])

                select_parts = [f"poh.`{poh_status_col}` AS status", f"poh.`{poh_vendor_col}` AS vendor_num"]
                join_part = ""

                if vnd_cols and vnd_vendor_col and vnd_lastname_col:
                    select_parts.append(f"COALESCE(vnd.`{vnd_lastname_col}`, CONCAT('Vendor ', poh.`{poh_vendor_col}`)) AS vendor_name")
                    join_part = f"LEFT JOIN vnd ON vnd.`{vnd_vendor_col}` = poh.`{poh_vendor_col}`"
                    use_vnd = True
                else:
                    select_parts.append(f"CONCAT('Vendor ', poh.`{poh_vendor_col}`) AS vendor_name")
                    use_vnd = False

                if poh_total_col:
                    select_parts.append(f"SUM(poh.`{poh_total_col}`) AS po_total")
                else:
                    select_parts.append("0 AS po_total")

                select_parts.append("COUNT(*) AS order_count")

                group_by_cols = f"poh.`{poh_status_col}`, poh.`{poh_vendor_col}`"
                if use_vnd and vnd_lastname_col:
                    group_by_cols += f", vnd.`{vnd_lastname_col}`"

                sql = f"""
                    SELECT {', '.join(select_parts)}
                    FROM poh
                    {join_part}
                    WHERE {' AND '.join(where_parts)}
                    GROUP BY {group_by_cols}
                    ORDER BY poh.`{poh_status_col}`, order_count DESC
                """

                rows = conn.execute(text(sql), params).mappings()
                status_map = {
                    3: "Complete",
                    4: "posted",
                    6: "received",
                    8: "open order",
                }

                # For each vendor, fetch individual order details from pod
                for r in rows:
                    vendor_num = str(r.get("vendor_num", ""))
                    status_code = int(r.get("status", 0))

                    # Fetch order details from pod for this vendor
                    order_details = []
                    if _table_exists(conn, db_name, "pod") and poh_id_col:
                        pod_cols = _get_columns(conn, db_name, "pod")
                        pod_po_col = _pick_column(pod_cols, ["po", "po_id", "poh_id", "invno"]) or poh_id_col
                        pod_total_col = _pick_column(pod_cols, ["total", "amount", "price"]) or None
                        pod_date_col = _pick_column(pod_cols, ["date", "pdate", "created_at"]) or None
                        pod_status_col = (
                            _pick_column(pod_cols, ["status", "stat"]) or None
                        )

                        # Get POs from poh for this vendor within PO date range and matching status
                        # Also get POH status for each order to compare with POD status
                        poh_ids_where = [f"poh.`{poh_vendor_col}` = :vnum", f"poh.`{poh_status_col}` = :stat"]
                        poh_ids_params = {"vnum": vendor_num, "stat": status_code}

                        if po_start_eff or po_end_eff:
                            if poh_rcv_col:
                                if po_start_eff:
                                    poh_ids_where.append(
                                        f"poh.`{poh_rcv_col}` >= :po_start_dt"
                                    )
                                if po_end_eff:
                                    poh_ids_where.append(
                                        f"poh.`{poh_rcv_col}` < :po_end_dt"
                                    )
                                poh_ids_params.update(
                                    {
                                        k: v
                                        for k, v in params.items()
                                        if k in ("po_start_dt", "po_end_dt")
                                    }
                                )

                        # Get POH order IDs and their statuses
                        poh_ids_sql = f"""
                            SELECT poh.`{poh_id_col}` AS po_id, poh.`{poh_status_col}` AS poh_status
                            FROM poh
                            WHERE {' AND '.join(poh_ids_where)}
                        """
                        poh_id_rows = conn.execute(text(poh_ids_sql), poh_ids_params).mappings()
                        # Create a map of po_id -> poh_status for comparison
                        poh_status_map = {
                            str(row.get("po_id")): int(row.get("poh_status", 0))
                            for row in poh_id_rows
                            if row.get("po_id")
                        }
                        po_ids = list(poh_status_map.keys())

                        if po_ids and pod_po_col:
                            pod_where = [f"`{pod_po_col}` IN :poids"]
                            pod_params = {"poids": tuple(po_ids)}

                            pod_select = [f"`{pod_po_col}` AS po_id"]
                            if pod_total_col:
                                pod_select.append(f"SUM(`{pod_total_col}`) AS order_total")
                            else:
                                pod_select.append("0 AS order_total")
                            if pod_date_col:
                                pod_select.append(f"MIN(DATE(`{pod_date_col}`)) AS order_date")
                            # Get POD status if available
                            if pod_status_col:
                                pod_select.append(
                                    f"MAX(`{pod_status_col}`) AS pod_status"
                                )

                            pod_sql = f"""
                                SELECT {', '.join(pod_select)}
                                FROM pod
                                WHERE {' AND '.join(pod_where)}
                                GROUP BY `{pod_po_col}`
                                ORDER BY `{pod_po_col}` DESC
                            """
                            pod_rows = conn.execute(text(pod_sql), pod_params).mappings()
                            for pod_row in pod_rows:
                                po_id_str = str(pod_row.get("po_id", ""))
                                poh_status_for_order = poh_status_map.get(
                                    po_id_str, status_code
                                )

                                # Get POD status if available, otherwise use POH status
                                pod_status_for_order = None
                                if (
                                    pod_status_col
                                    and pod_row.get("pod_status") is not None
                                ):
                                    try:
                                        pod_status_for_order = int(
                                            pod_row.get("pod_status", 0)
                                        )
                                    except (ValueError, TypeError):
                                        pod_status_for_order = None

                                # Use POH status as default if POD status differs or is not available
                                final_status = poh_status_for_order
                                if (
                                    pod_status_for_order is not None
                                    and pod_status_for_order != poh_status_for_order
                                ):
                                    # POD status differs from POH status - use POH status as default
                                    final_status = poh_status_for_order

                                order_details.append(
                                    {
                                        "po_id": po_id_str,
                                        "order_total": float(
                                            pod_row.get("order_total", 0) or 0
                                        ),
                                        "order_date": (
                                            str(pod_row.get("order_date", ""))
                                            if pod_row.get("order_date")
                                            else None
                                        ),
                                        "status": status_map.get(
                                            final_status, str(final_status)
                                        ),
                                    }
                                )
                                freshness.track(pod_row.get("order_date"))

                    purchase_orders.append({
                        "vendor_name": str(r.get("vendor_name", "")),
                        "vendor_num": vendor_num,
                        "status": status_map.get(status_code, str(status_code)),
                        "order_count": int(r.get("order_count", 0) or 0),
                        "po_total": float(r.get("po_total", 0) or 0),
                        "orders": order_details,
                    })

            # Inventory value via inv lcost/acost * onhand
            inventory = None
            if _table_exists(conn, db_name, "inv"):
                inv_cols = _get_columns(conn, db_name, "inv")
                qty_col = _pick_column(inv_cols, ["onhand", "qty", "stock"]) or None
                lcost_col = _pick_column(inv_cols, ["lcost", "last_cost"]) or None
                acost_col = _pick_column(inv_cols, ["acost", "avg_cost"]) or None
                inv_cat_col = _pick_column(inv_cols, ["cat", "category", "catcode"]) or None

                if qty_col and (lcost_col or acost_col):
                    parts = []
                    if lcost_col:
                        parts.append(f"SUM(COALESCE(`{qty_col}`,0) * COALESCE(`{lcost_col}`,0)) AS total_lcost")
                    if acost_col:
                        parts.append(f"SUM(COALESCE(`{qty_col}`,0) * COALESCE(`{acost_col}`,0)) AS total_acost")

                    # Try to get category breakdown if cat table exists
                    segments = []
                    if inv_cat_col and _table_exists(conn, db_name, "cat"):
                        cat_cols = _get_columns(conn, db_name, "cat")
                        cat_code = _pick_column(cat_cols, ["cat", "code", "id"]) or "cat"
                        cat_label = _pick_column(cat_cols, ["desc", "description", "name", "label"]) or cat_code

                        cost_expr = f"SUM(COALESCE(inv.`{qty_col}`,0) * COALESCE(inv.`{lcost_col or acost_col}`,0))"
                        cat_sql = f"""
                            SELECT COALESCE(cat.`{cat_label}`, inv.`{inv_cat_col}`) AS category_label,
                                   {cost_expr} AS category_value
                            FROM inv
                            LEFT JOIN cat ON cat.`{cat_code}` = inv.`{inv_cat_col}`
                            WHERE COALESCE(inv.`{qty_col}`,0) > 0
                            GROUP BY COALESCE(cat.`{cat_label}`, inv.`{inv_cat_col}`)
                            ORDER BY category_value DESC
                            LIMIT 10
                        """
                        cat_rows = conn.execute(text(cat_sql)).mappings()
                        segments = [
                            {
                                "label": str(r["category_label"]) if r["category_label"] else "Unknown",
                                "value": float(r["category_value"] or 0)
                            }
                            for r in cat_rows
                        ]
                    elif inv_cat_col:
                        # No cat table, but we have category column in inv
                        cost_expr = f"SUM(COALESCE(`{qty_col}`,0) * COALESCE(`{lcost_col or acost_col}`,0))"
                        cat_sql = f"""
                            SELECT `{inv_cat_col}` AS category_label,
                                   {cost_expr} AS category_value
                            FROM inv
                            WHERE COALESCE(`{qty_col}`,0) > 0
                            GROUP BY `{inv_cat_col}`
                            ORDER BY category_value DESC
                            LIMIT 10
                        """
                        cat_rows = conn.execute(text(cat_sql)).mappings()
                        segments = [
                            {
                                "label": str(r["category_label"]) if r["category_label"] else "Unknown",
                                "value": float(r["category_value"] or 0)
                            }
                            for r in cat_rows
                        ]

                    inv_sql = f"SELECT {', '.join(parts)} FROM inv"
                    r = conn.execute(text(inv_sql)).mappings().first()
                    inventory = {
                        "total_value": float((r or {}).get("total_lcost", 0) or 0),
                        "segments": segments,
                    }

            # Sales history by year from hst
            history: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "hst"):
                hst_cols = _get_columns(conn, db_name, "hst")
                hst_date = _pick_column(hst_cols, ["date", "tdate", "sale_date", "trans_date"]) or "date"
                hst_qty = _pick_column(hst_cols, ["qty", "quantity"]) or None
                hst_price = _pick_column(hst_cols, ["amount", "price", "total"]) or None
                hst_sku = _pick_column(hst_cols, ["sku"]) or "sku"
                amount_expr_hist = None
                if hst_price and hst_qty:
                    amount_expr_hist = f"SUM(`{hst_qty}` * `{hst_price}`)"
                elif hst_price:
                    amount_expr_hist = f"SUM(`{hst_price}`)"
                else:
                    amount_expr_hist = "SUM(0)"
                where_parts = ["1=1"]
                params7: Dict[str, Any] = {}
                if start:
                    where_parts.append(f"DATE(`{hst_date}`) >= :h2_start")
                    params7["h2_start"] = start
                if end:
                    where_parts.append(f"DATE(`{hst_date}`) <= :h2_end")
                    params7["h2_end"] = end
                hist_sql = f"""
                    SELECT `{hst_sku}` AS sku, YEAR(`{hst_date}`) AS year, {amount_expr_hist} AS total
                    FROM hst
                    WHERE {' AND '.join(where_parts)}
                    GROUP BY `{hst_sku}`, YEAR(`{hst_date}`)
                    ORDER BY year DESC, total DESC
                    LIMIT 200
                """
                rows = conn.execute(text(hist_sql), params7).mappings()
                # join inv for product_name
                inv_names: Dict[str, str] = {}
                skus = list({str(r["sku"]) for r in rows})
                if skus and _table_exists(conn, db_name, "inv"):
                    inv_cols = _get_columns(conn, db_name, "inv")
                    inv_sku = _pick_column(inv_cols, ["sku"]) or "sku"
                    inv_name = _pick_column(inv_cols, ["desc", "description", "name"]) or inv_sku
                    in_params = {"skus": tuple(skus)}
                    inv_sql = text(f"SELECT `{inv_sku}` AS sku, `{inv_name}` AS name FROM inv WHERE `{inv_sku}` IN :skus").bindparams()
                    inv_rows = conn.execute(inv_sql, in_params).mappings()
                    for r2 in inv_rows:
                        inv_names[str(r2["sku"])] = str(r2["name"]) if r2["name"] is not None else ""
                history = [
                    {
                        "product_name": inv_names.get(str(r["sku"]), ""),
                        "sku": str(r["sku"]),
                        "year": int(r["year"]),
                        "total": float(r["total"] or 0),
                    }
                    for r in rows
                ]
                for r in rows:
                    try:
                        freshness.track(datetime(int(r["year"]), 1, 1))
                    except Exception:
                        continue

            payload = {
                "store": response_meta,
                "sales": sales_data,
                "summary": summary,
                "top_items": top_items,
                "breakdowns": {
                    "hourly": hourly,
                    "payment_methods": payment_methods,
                    "categories": categories,
                },
                "purchase_orders": purchase_orders,
                "inventory": inventory,
                "sales_history": history,
            }

            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info

            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")


@router.get("/operations")
def get_operations_insights(
    store: Optional[str] = Query(
        default=None, description="Store identifier (store_id or store_db)"
    ),
    user: dict = Depends(get_auth_user),
):
    """Get store operations insights: overall system-wide metrics including transaction counts, gift cards, products, and supplier invoices. All metrics show all-time totals without date filtering."""
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        with engine.connect() as conn:
            freshness = FreshnessTracker()
            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            # Operations tab: transaction/till/gift cards show current month up to now, products/supplier invoices show overall
            # Calculate current month range (first day 00:00:00 to current date/time)
            from datetime import datetime, timedelta

            now = datetime.now()
            current_month_start = datetime(now.year, now.month, 1)
            # Use current date/time as end (not end of month) - only count transactions up to now
            current_month_end = now

            month_start_str = current_month_start.strftime("%Y-%m-%d 00:00:00")
            month_end_str = current_month_end.strftime("%Y-%m-%d %H:%M:%S")

            # Transaction counts and till amounts from jnl (current month only)
            tx_count = 0
            total_till = 0.0
            gift_redeem_count = 0
            gift_redeem_amount = 0.0
            gift_purchase_count = 0
            gift_purchase_amount = 0.0

            if _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_date_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_sale_col = _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or "sale"
                jnl_rflag_col = _pick_column(jnl_cols, ["rflag"]) or None
                jnl_price_col = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_descript_col = _pick_column(jnl_cols, ["descript", "description", "desc"]) or None

                if jnl_line_col and jnl_sale_col and jnl_date_col:
                    # Get valid transactions for current month up to current date/time (first day 00:00:00 to now)
                    where_parts = []
                    params: Dict[str, Any] = {}
                    if jnl_rflag_col:
                        where_parts.append(f"`{jnl_rflag_col}` = 0")
                    # Filter by current month up to current date/time
                    # Use DATE() on column to handle both DATE and DATETIME columns properly
                    where_parts.append(f"DATE(`{jnl_date_col}`) >= :month_start")
                    where_parts.append(f"DATE(`{jnl_date_col}`) <= :month_end")
                    # Extract just the date part for comparison (YYYY-MM-DD)
                    params["month_start"] = month_start_str.split()[
                        0
                    ]  # Get date part only
                    params["month_end"] = month_end_str.split()[0]  # Get date part only

                    # Get all rows for grouping by sale (current month only)
                    sql = f"""
                        SELECT `{jnl_sale_col}` AS sale_id, `{jnl_line_col}` AS line_code,
                               `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript
                        FROM jnl
                        WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                        ORDER BY `{jnl_sale_col}`, `{jnl_line_col}`
                    """
                    rows = conn.execute(text(sql), params).mappings()

                    # Group by sale_id to find valid transactions
                    # Valid transaction = sale has both line 950 (till) and line 980 (tender)
                    current_sale = None
                    current_group = []
                    for row in rows:
                        sale_id = str(row.get("sale_id", ""))
                        line_code = str(row.get("line_code", ""))

                        if sale_id != current_sale:
                            # Process previous group
                            if current_group:
                                # Convert line codes to strings for consistent comparison
                                lines = {
                                    str(r.get("line_code", "")) for r in current_group
                                }
                                # Valid transaction must have both till line (950) and tender line (980)
                                if "950" in lines and "980" in lines:
                                    tx_count += 1
                                    # Sum all till amounts from line 950 for this transaction
                                    for r in current_group:
                                        if str(r.get("line_code", "")) == "950":
                                            try:
                                                price_val = float(r.get("price") or 0)
                                                # Sanity check to exclude outliers
                                                if abs(price_val) <= 100000:
                                                    total_till += price_val
                                            except:
                                                pass

                                    # Check for gift card patterns in line 980 (tender line)
                                    if jnl_descript_col:
                                        has_gift_pattern = False
                                        for r in current_group:
                                            if str(r.get("line_code", "")) == "980":
                                                desc = str(r.get("descript", "")).lower()
                                                if any(pattern in desc for pattern in ["gift card", "giftcard", "gc", "gift cert", "giftcert", "gift certificate", "giftcertificate", "gift crd", "giftcrd"]):
                                                    has_gift_pattern = True
                                                    break

                                        if has_gift_pattern:
                                            # Check if it's a redemption (negative price) or purchase (positive)
                                            for r in current_group:
                                                if str(r.get("line_code", "")) == "950":
                                                    try:
                                                        price_val = float(r.get("price") or 0)
                                                        if price_val < 0:
                                                            gift_redeem_count += 1
                                                            gift_redeem_amount += abs(price_val)
                                                        elif price_val > 0:
                                                            gift_purchase_count += 1
                                                            gift_purchase_amount += price_val
                                                    except:
                                                        pass

                            current_sale = sale_id
                            current_group = [row]
                        else:
                            current_group.append(row)

                    # Process last group
                    if current_group:
                        # Convert line codes to strings for consistent comparison
                        lines = {str(r.get("line_code", "")) for r in current_group}
                        # Valid transaction must have both till line (950) and tender line (980)
                        if "950" in lines and "980" in lines:
                            tx_count += 1
                            # Sum all till amounts from line 950 for this transaction
                            for r in current_group:
                                if str(r.get("line_code", "")) == "950":
                                    try:
                                        price_val = float(r.get("price") or 0)
                                        # Sanity check to exclude outliers
                                        if abs(price_val) <= 100000:
                                            total_till += price_val
                                    except:
                                        pass

            # Product counts from inv (overall view - no date filtering)
            product_count = 0
            first_product_date = None
            last_product_date = None
            if _table_exists(conn, db_name, "inv"):
                inv_cols = _get_columns(conn, db_name, "inv")
                inv_deleted_col = _pick_column(inv_cols, ["deleted", "del", "is_deleted"]) or None
                inv_cdate_col = _pick_column(inv_cols, ["cdate", "created_date", "created_at", "date"]) or None

                where_parts = []
                if inv_deleted_col:
                    where_parts.append(f"UPPER(`{inv_deleted_col}`) != 'T'")

                # Get overall product count and date range (no date filtering)
                if inv_cdate_col:
                    sql = f"""
                        SELECT COUNT(*) AS cnt,
                               MIN(DATE(`{inv_cdate_col}`)) AS first_date,
                               MAX(DATE(`{inv_cdate_col}`)) AS last_date
                        FROM inv
                        {'WHERE ' + ' AND '.join(where_parts) if where_parts else ''}
                    """
                else:
                    sql = f"""
                        SELECT COUNT(*) AS cnt
                        FROM inv
                        {'WHERE ' + ' AND '.join(where_parts) if where_parts else ''}
                    """
                r = conn.execute(text(sql)).mappings().first()
                if r:
                    product_count = int(r.get("cnt", 0) or 0)
                    if inv_cdate_col:
                        first_product_date = str(r.get("first_date", "")) if r.get("first_date") else None
                        last_product_date = str(r.get("last_date", "")) if r.get("last_date") else None
                        freshness.track(first_product_date, last_product_date)

            # Supplier invoice counts from poh (overall view - only posted status 4, no date filtering)
            supplier_invoice_count = 0
            first_supplier_date = None
            last_supplier_date = None
            if _table_exists(conn, db_name, "poh"):
                poh_cols = _get_columns(conn, db_name, "poh")
                poh_status_col = _pick_column(poh_cols, ["status", "stat"]) or "status"
                poh_vendor_col = _pick_column(poh_cols, ["vendor", "vcode"]) or "vendor"
                poh_rcvdate_col = _pick_column(poh_cols, ["rcvdate", "received_date", "rcv_date"]) or None

                # Only posted invoices (status 4), exclude test vendors, no date filtering
                where_parts = [
                    f"`{poh_status_col}` = 4",
                    f"`{poh_vendor_col}` NOT IN ('9998', '9999', 9998, 9999)",
                ]

                if poh_rcvdate_col:
                    sql = f"""
                        SELECT COUNT(*) AS cnt,
                               MIN(DATE(`{poh_rcvdate_col}`)) AS first_date,
                               MAX(DATE(`{poh_rcvdate_col}`)) AS last_date
                        FROM poh
                        WHERE {' AND '.join(where_parts)}
                    """
                    r = conn.execute(text(sql)).mappings().first()
                    if r:
                        supplier_invoice_count = int(r.get("cnt", 0) or 0)
                        first_supplier_date = str(r.get("first_date", "")) if r.get("first_date") else None
                        last_supplier_date = str(r.get("last_date", "")) if r.get("last_date") else None
                        freshness.track(first_supplier_date, last_supplier_date)
                else:
                    sql = f"""
                        SELECT COUNT(*) AS cnt
                        FROM poh
                        WHERE {' AND '.join(where_parts)}
                    """
                    r = conn.execute(text(sql)).mappings().first()
                    if r:
                        supplier_invoice_count = int(r.get("cnt", 0) or 0)

            payload = {
                "store": response_meta,
                "transaction_count": tx_count,
                "total_till": round(total_till, 2),
                "gift_cards": {
                    "redemptions": {
                        "count": gift_redeem_count,
                        "amount": round(gift_redeem_amount, 2),
                    },
                    "purchases": {
                        "count": gift_purchase_count,
                        "amount": round(gift_purchase_amount, 2),
                    },
                },
                "products": {
                    "count": product_count,
                    "first_created": first_product_date,
                    "last_created": last_product_date,
                },
                "supplier_invoices": {
                    "count": supplier_invoice_count,
                    "first_date": first_supplier_date,
                    "last_date": last_supplier_date,
                },
            }

            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info
            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get operations insights: {str(e)}")


@router.get("/tenders")
def get_tenders_insights(
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    user: dict = Depends(get_auth_user),
):
    """Get tender/payment method insights with sale vs reversal breakdown."""
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        with engine.connect() as conn:
            freshness = FreshnessTracker()
            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            if not start:
                from datetime import timedelta
                start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')

            tenders_data = []
            daily_trends = []

            # Use same logic as payment methods in sales insights
            # Join to jnh, filter by jnh.tstamp, exclude void sales, handle cash change
            if _table_exists(conn, db_name, "jnl") and _table_exists(
                conn, db_name, "jnh"
            ):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_rflag = "rflag" if "rflag" in jnl_cols else None
                cat_col = "cat" if "cat" in jnl_cols else None
                descript_col = (
                    _pick_column(jnl_cols, ["descript", "description", "desc"]) or None
                )
                amt_col = None
                for c in ["amount", "total", "price"]:
                    if c in jnl_cols:
                        amt_col = c
                        break

                jnl_sale_col = (
                    _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or "sale"
                )
                jnl_sku_col = (
                    _pick_column(jnl_cols, ["sku", "item", "item_id"]) or "sku"
                )

                # Get jnh columns for date filtering
                jnh_cols = _get_columns(conn, db_name, "jnh")
                jnh_time = (
                    _pick_column(jnh_cols, ["tstamp", "timestamp", "time", "t_time"])
                    or "tstamp"
                )
                jnh_sale = (
                    _pick_column(jnh_cols, ["sale", "sale_id", "invno"]) or "sale"
                )

                if jnl_line_col and amt_col:
                    where_parts = []
                    params: Dict[str, Any] = {}

                    # Date filter: use jnh.tstamp to match sales insights logic
                    if start:
                        where_parts.append(f"jnh.`{jnh_time}` >= :start_dt")
                        params["start_dt"] = f"{start} 00:00:00"
                    if end:
                        try:
                            from datetime import timedelta

                            end_next = (
                                datetime.fromisoformat(end) + timedelta(days=1)
                            ).strftime("%Y-%m-%d 00:00:00")
                        except Exception:
                            end_next = f"{end} 23:59:59"
                        where_parts.append(f"jnh.`{jnh_time}` < :end_dt")
                        params["end_dt"] = end_next

                    # Filter: RFLAG <= 0 (excludes returns/voids, includes all valid tenders)
                    if jnl_rflag:
                        where_parts.append(f"jnl.`{jnl_rflag}` <= 0")

                    # Exclude void sales: sales where any SKU line (sku > 0) has rflag = 4
                    void_sales_where = []
                    if jnl_rflag:
                        void_sales_where.append(f"jnl_void.`{jnl_rflag}` = 4")
                    void_sales_where.append(f"jnl_void.`{jnl_sku_col}` > 0")

                    # Build payment type identification and cash change logic
                    # For ecommerce sales, cat may be in promo column, so use promo as fallback
                    promo_col = "promo" if "promo" in jnl_cols else None
                    if cat_col and _table_exists(conn, db_name, "cat"):
                        cat_table_cols = _get_columns(conn, db_name, "cat")
                        cat_code_col = (
                            _pick_column(cat_table_cols, ["cat", "code", "id"]) or "cat"
                        )
                        cat_name_col = (
                            _pick_column(
                                cat_table_cols, ["name", "desc", "description", "label"]
                            )
                            or "name"
                        )

                        # Build cat lookup: use cat column, fallback to promo if cat is empty/0
                        # This handles ecommerce sales where cat might be in promo column
                        cat_lookup_expr = (
                            f"COALESCE(NULLIF(jnl.`{cat_col}`, 0), NULLIF(jnl.`{promo_col}`, 0), 0)"
                            if promo_col
                            else f"COALESCE(NULLIF(jnl.`{cat_col}`, 0), 0)"
                        )

                        # Cash change subquery
                        cash_change_where = [f"jnl_cc.`{jnl_line_col}` = 999"]
                        if jnl_rflag:
                            cash_change_where.append(f"jnl_cc.`{jnl_rflag}` <= 0")
                        if start:
                            cash_change_where.append(
                                f"jnh_cc.`{jnh_time}` >= :cc_start_dt"
                            )
                            params["cc_start_dt"] = params["start_dt"]
                        if end:
                            cash_change_where.append(
                                f"jnh_cc.`{jnh_time}` < :cc_end_dt"
                            )
                            params["cc_end_dt"] = params["end_dt"]

                        cash_change_void_where = []
                        if jnl_rflag:
                            cash_change_void_where.append(
                                f"jnl_void_cc.`{jnl_rflag}` = 4"
                            )
                        cash_change_void_where.append(
                            f"jnl_void_cc.`{jnl_sku_col}` > 0"
                        )

                        cash_tender_check_where = [
                            f"jnl_cash_tender.`{jnl_line_col}` BETWEEN 980 AND 989"
                        ]
                        if jnl_rflag:
                            cash_tender_check_where.append(
                                f"jnl_cash_tender.`{jnl_rflag}` <= 0"
                            )

                        cash_tender_identify = [
                            f"UPPER(COALESCE(cat_cash_tender.`{cat_name_col}`, '')) LIKE '%CASH%'"
                        ]
                        if descript_col:
                            cash_tender_identify.append(
                                f"UPPER(jnl_cash_tender.`{descript_col}`) LIKE '%CASH%'"
                            )
                        cash_tender_identify_condition = " OR ".join(
                            cash_tender_identify
                        )

                        # Main query: match payment methods logic from sales insights
                        sql = f"""
                            SELECT 
                                DATE(jnh.`{jnh_time}`) AS date,
                                COALESCE(
                                    (SELECT cat.`{cat_name_col}` FROM cat WHERE cat.`{cat_code_col}` = {cat_lookup_expr} LIMIT 1),
                                    {f"jnl.`{descript_col}`" if descript_col else "NULL"},
                                    CAST({cat_lookup_expr} AS CHAR)
                                ) AS payment_type,
                                CASE 
                                    WHEN (
                                        UPPER(COALESCE((SELECT cat.`{cat_name_col}` FROM cat WHERE cat.`{cat_code_col}` = {cat_lookup_expr} LIMIT 1), '')) LIKE '%CASH%'
                                        {' OR UPPER(jnl.`' + descript_col + '`) LIKE \'%CASH%\'' if descript_col else ''}
                                    ) THEN 
                                        GREATEST(0, jnl.`{amt_col}` - ABS(COALESCE(cash_change_per_sale.cash_change, 0)))
                                    ELSE 
                                        jnl.`{amt_col}`
                                END AS amount,
                                jnl.`{amt_col}` AS raw_amount,
                                jnh.`{jnh_sale}` AS sale_id
                            FROM jnl
                            INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                            LEFT JOIN (
                                SELECT 
                                    jnl_cc.`{jnl_sale_col}` AS sale_id,
                                    SUM(jnl_cc.`{amt_col}`) AS cash_change
                                FROM jnl jnl_cc
                                INNER JOIN jnh jnh_cc ON jnh_cc.`{jnh_sale}` = jnl_cc.`{jnl_sale_col}`
                                WHERE {' AND '.join(cash_change_where)}
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM jnl jnl_void_cc
                                      WHERE jnl_void_cc.`{jnl_sale_col}` = jnl_cc.`{jnl_sale_col}`
                                        AND {' AND '.join(cash_change_void_where)}
                                      LIMIT 1
                                  )
                                  AND EXISTS (
                                      SELECT 1
                                      FROM jnl jnl_cash_tender
                                      LEFT JOIN cat cat_cash_tender ON cat_cash_tender.`{cat_code_col}` = jnl_cash_tender.`{cat_col}`
                                      WHERE jnl_cash_tender.`{jnl_sale_col}` = jnl_cc.`{jnl_sale_col}`
                                        AND {' AND '.join(cash_tender_check_where)}
                                        AND ({cash_tender_identify_condition})
                                      LIMIT 1
                                  )
                                GROUP BY jnl_cc.`{jnl_sale_col}`
                            ) AS cash_change_per_sale ON cash_change_per_sale.sale_id = jnl.`{jnl_sale_col}`
                            WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                              AND jnl.`{jnl_line_col}` BETWEEN 980 AND 989
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM jnl jnl_void
                                  WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                    AND {' AND '.join(void_sales_where)}
                                  LIMIT 1
                              )
                            ORDER BY DATE(jnh.`{jnh_time}`), payment_type
                        """
                    else:
                        # Fallback: no cat table, use descript
                        sql = f"""
                            SELECT 
                                DATE(jnh.`{jnh_time}`) AS date,
                                {f"jnl.`{descript_col}`" if descript_col else "CAST(jnl.`{jnl_line_col}` AS CHAR)"} AS payment_type,
                                jnl.`{amt_col}` AS amount,
                                jnl.`{amt_col}` AS raw_amount,
                                jnh.`{jnh_sale}` AS sale_id
                            FROM jnl
                            INNER JOIN jnh ON jnh.`{jnh_sale}` = jnl.`{jnl_sale_col}`
                            WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                              AND jnl.`{jnl_line_col}` BETWEEN 980 AND 989
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM jnl jnl_void
                                  WHERE jnl_void.`{jnl_sale_col}` = jnl.`{jnl_sale_col}`
                                    AND {' AND '.join(void_sales_where)}
                                  LIMIT 1
                              )
                            ORDER BY DATE(jnh.`{jnh_time}`), payment_type
                        """

                    rows = conn.execute(text(sql), params).mappings()

                    # Group by date and payment type
                    from collections import defaultdict
                    by_type = defaultdict(
                        lambda: {
                            "sale_amount": 0.0,
                            "sale_count": set(),
                            "reversal_amount": 0.0,
                            "reversal_count": set(),
                        }
                    )
                    by_date_type = defaultdict(
                        lambda: {
                            "sale_amount": 0.0,
                            "sale_count": set(),
                            "reversal_amount": 0.0,
                            "reversal_count": set(),
                        }
                    )

                    for row in rows:
                        try:
                            amount = float(row.get("amount") or 0)
                            raw_amount = float(row.get("raw_amount") or 0)
                            payment_type = str(row.get("payment_type") or "Unknown")
                            date_str = str(row.get("date", "")) if row.get("date") else ""
                            sale_id = row.get("sale_id")

                            if date_str:
                                freshness.track(date_str)

                            # Normalize payment type name
                            payment_type = _normalize_payment_type(payment_type)

                            if raw_amount >= 0:
                                by_type[payment_type]["sale_amount"] += amount
                                if sale_id:
                                    by_type[payment_type]["sale_count"].add(sale_id)
                                if date_str:
                                    by_date_type[(date_str, payment_type)][
                                        "sale_amount"
                                    ] += amount
                                    if sale_id:
                                        by_date_type[(date_str, payment_type)][
                                            "sale_count"
                                        ].add(sale_id)
                            else:
                                by_type[payment_type]["reversal_amount"] += abs(amount)
                                if sale_id:
                                    by_type[payment_type]["reversal_count"].add(sale_id)
                                if date_str:
                                    by_date_type[(date_str, payment_type)][
                                        "reversal_amount"
                                    ] += abs(amount)
                                    if sale_id:
                                        by_date_type[(date_str, payment_type)][
                                            "reversal_count"
                                        ].add(sale_id)
                        except:
                            continue

                    # Build tenders summary
                    for payment_type, stats in by_type.items():
                        tenders_data.append(
                            {
                                "payment_type": payment_type,
                                "sale_amount": round(stats["sale_amount"], 2),
                                "sale_count": len(stats["sale_count"]),
                                "reversal_amount": round(stats["reversal_amount"], 2),
                                "reversal_count": len(stats["reversal_count"]),
                                "net_amount": round(
                                    stats["sale_amount"] - stats["reversal_amount"], 2
                                ),
                            }
                        )

                    # Build daily trends
                    daily_by_date = defaultdict(lambda: defaultdict(lambda: {"sale_amount": 0.0, "reversal_amount": 0.0}))
                    for (date_str, payment_type), stats in by_date_type.items():
                        daily_by_date[date_str][payment_type] = {
                            "sale_amount": stats["sale_amount"],
                            "reversal_amount": stats["reversal_amount"]
                        }

                    for date_str in sorted(daily_by_date.keys()):
                        daily_trends.append({
                            "date": date_str,
                            "by_type": {
                                pt: {
                                    "sale_amount": round(stats["sale_amount"], 2),
                                    "reversal_amount": round(stats["reversal_amount"], 2)
                                }
                                for pt, stats in daily_by_date[date_str].items()
                            }
                        })

            payload = {
                "store": response_meta,
                "tenders": sorted(
                    tenders_data, key=lambda x: x["net_amount"], reverse=True
                ),
                "daily_trends": daily_trends,
            }
            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info
            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tenders insights: {str(e)}")


@router.get("/gateway")
def get_gateway_insights(
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    user: dict = Depends(get_auth_user),
):
    """Get credit card and gateway insights with ecommerce vs in-store breakdown."""
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        with engine.connect() as conn:
            freshness = FreshnessTracker()
            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            if not start:
                from datetime import timedelta
                start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')

            credit_card_data = []
            total_cc_sales = 0.0
            total_ecom_sales = 0.0

            if _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_date_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_price_col = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_descript_col = _pick_column(jnl_cols, ["descript", "description", "desc"]) or None

                if jnl_date_col and jnl_line_col and jnl_price_col and jnl_descript_col:
                    where_parts = [f"`{jnl_line_col}` BETWEEN 980 AND 989"]
                    params: Dict[str, Any] = {}
                    if start:
                        where_parts.append(f"DATE(`{jnl_date_col}`) >= :start")
                        params["start"] = start
                    if end:
                        where_parts.append(f"DATE(`{jnl_date_col}`) <= :end")
                        params["end"] = end

                    sql = f"""
                        SELECT `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript,
                               DATE(`{jnl_date_col}`) AS txn_date
                        FROM jnl
                        WHERE {' AND '.join(where_parts)}
                    """
                    rows = conn.execute(text(sql), params).mappings()

                    from collections import defaultdict
                    by_type_ecom = defaultdict(lambda: {"in_store": 0.0, "ecommerce": 0.0})

                    for row in rows:
                        try:
                            price = float(row.get("price") or 0)
                            if price < 0:
                                continue  # Skip reversals
                            freshness.track(row.get("txn_date"))

                            descript = str(row.get("descript", "") or "").lower()

                            # Check if it's a credit card
                            payment_type = _normalize_payment_type(descript)
                            if payment_type in ["Visa", "MasterCard", "AmEx", "Discover", "Debit"]:
                                # Check if ecommerce
                                is_ecom = any(x in descript for x in ["web", "ecommerce", "online", "ecom", "e-commerce", "internet", "online order", "web order", "online sale"])

                                if is_ecom:
                                    by_type_ecom[payment_type]["ecommerce"] += price
                                    total_ecom_sales += price
                                else:
                                    by_type_ecom[payment_type]["in_store"] += price
                                    total_cc_sales += price
                        except:
                            continue

                    for payment_type, stats in by_type_ecom.items():
                        credit_card_data.append({
                            "tender_type": payment_type,
                            "in_store_sales": round(stats["in_store"], 2),
                            "ecommerce_sales": round(stats["ecommerce"], 2),
                            "total_sales": round(stats["in_store"] + stats["ecommerce"], 2)
                        })

            payload = {
                "store": response_meta,
                "credit_cards": sorted(
                    credit_card_data, key=lambda x: x["total_sales"], reverse=True
                ),
                "summary": {
                    "total_credit_card_sales": round(total_cc_sales, 2),
                    "total_ecommerce_sales": round(total_ecom_sales, 2),
                    "total_combined": round(total_cc_sales + total_ecom_sales, 2),
                },
            }
            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info
            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get gateway insights: {str(e)}")


@router.get("/quick")
def get_quick_insights(
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    user: dict = Depends(get_auth_user),
):
    """Get quick insights for today: sales summary, transaction count, top items, payment methods, gift cards."""
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        today = datetime.now().strftime('%Y-%m-%d')

        with engine.connect() as conn:
            freshness = FreshnessTracker()
            freshness.track(today)
            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            # Get today's sales from sales endpoint logic
            detected = _detect_sales_source(conn, db_name)
            today_sales = 0.0
            today_items = 0
            today_tx_count = 0

            if detected and _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_date_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_sale_col = _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or "sale"
                jnl_price_col = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_rflag_col = _pick_column(jnl_cols, ["rflag"]) or None

                if jnl_date_col and jnl_line_col and jnl_sale_col:
                    where_parts = [f"DATE(`{jnl_date_col}`) = :today"]
                    if jnl_rflag_col:
                        where_parts.append(f"`{jnl_rflag_col}` = 0")

                    sql = f"""
                        SELECT `{jnl_sale_col}` AS sale_id, `{jnl_line_col}` AS line_code,
                               `{jnl_price_col}` AS price
                        FROM jnl
                        WHERE {' AND '.join(where_parts)}
                        ORDER BY `{jnl_sale_col}`, `{jnl_line_col}`
                    """
                    rows = conn.execute(text(sql), {"today": today}).mappings()

                    current_sale = None
                    current_group = []
                    for row in rows:
                        sale_id = str(row.get("sale_id", ""))
                        if sale_id != current_sale:
                            if current_group:
                                lines = {r.get("line_code", "") for r in current_group}
                                if "950" in lines and "980" in lines:
                                    today_tx_count += 1
                                    for r in current_group:
                                        if r.get("line_code") == "950":
                                            try:
                                                price_val = float(r.get("price") or 0)
                                                if abs(price_val) <= 100000:
                                                    today_sales += price_val
                                            except:
                                                pass
                            current_sale = sale_id
                            current_group = [row]
                        else:
                            current_group.append(row)

                    if current_group:
                        lines = {r.get("line_code", "") for r in current_group}
                        if "950" in lines and "980" in lines:
                            today_tx_count += 1
                            for r in current_group:
                                if r.get("line_code") == "950":
                                    try:
                                        price_val = float(r.get("price") or 0)
                                        if abs(price_val) <= 100000:
                                            today_sales += price_val
                                    except:
                                        pass

            # Today's top 5 items
            top_items_today = []
            if _table_exists(conn, db_name, "hst"):
                hst_cols = _get_columns(conn, db_name, "hst")
                hst_date = _pick_column(hst_cols, ["date", "tdate", "sale_date", "trans_date"]) or "date"
                hst_qty = _pick_column(hst_cols, ["qty", "quantity"]) or "qty"
                hst_pack = _pick_column(hst_cols, ["pack", "mult", "casepack"]) or None
                hst_sku = _pick_column(hst_cols, ["sku"]) or "sku"

                if hst_pack in hst_cols:
                    qty_expr_today = f"SUM(COALESCE(`{hst_qty}`,0) / NULLIF(COALESCE(`{hst_pack}`,1),0))"
                else:
                    qty_expr_today = f"SUM(COALESCE(`{hst_qty}`,0))"

                sql = f"""
                    SELECT `{hst_sku}` AS sku, {qty_expr_today} AS total_qty
                    FROM hst
                    WHERE DATE(`{hst_date}`) = :today AND `{hst_sku}` > 0
                    GROUP BY `{hst_sku}`
                    ORDER BY total_qty DESC
                    LIMIT 5
                """
                rows = conn.execute(text(sql), {"today": today}).mappings()
                skus = [str(r["sku"]) for r in rows]

                inv_names = {}
                if skus and _table_exists(conn, db_name, "inv"):
                    inv_cols = _get_columns(conn, db_name, "inv")
                    inv_sku = _pick_column(inv_cols, ["sku"]) or "sku"
                    inv_name = _pick_column(inv_cols, ["desc", "description", "name"]) or inv_sku
                    in_params = {"skus": tuple(skus)}
                    inv_sql = text(f"SELECT `{inv_sku}` AS sku, `{inv_name}` AS name FROM inv WHERE `{inv_sku}` IN :skus").bindparams()
                    inv_rows = conn.execute(inv_sql, in_params).mappings()
                    for r in inv_rows:
                        inv_names[str(r["sku"])] = str(r["name"]) if r["name"] is not None else ""

                top_items_today = [
                    {
                        "sku": str(r["sku"]),
                        "description": inv_names.get(str(r["sku"]), ""),
                        "quantity": float(r["total_qty"])
                    }
                    for r in rows
                ]

            # Today's payment methods
            payment_methods_today = []
            if _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_date_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_price_col = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_descript_col = _pick_column(jnl_cols, ["descript", "description", "desc"]) or None

                if jnl_date_col and jnl_line_col and jnl_price_col and jnl_descript_col:
                    sql = f"""
                        SELECT `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript
                        FROM jnl
                        WHERE DATE(`{jnl_date_col}`) = :today 
                          AND `{jnl_line_col}` BETWEEN 980 AND 989
                    """
                    rows = conn.execute(text(sql), {"today": today}).mappings()

                    from collections import defaultdict
                    by_type = defaultdict(lambda: 0.0)
                    for row in rows:
                        try:
                            price = float(row.get("price") or 0)
                            if price >= 0:
                                descript = str(row.get("descript", "") or "")
                                payment_type = _normalize_payment_type(descript)
                                by_type[payment_type] += price
                        except:
                            continue

                    payment_methods_today = [
                        {"method": pt, "amount": round(amt, 2)}
                        for pt, amt in sorted(by_type.items(), key=lambda x: x[1], reverse=True)
                    ]

            # Today's gift card activity
            gift_card_activity = {"redemptions": 0, "purchases": 0, "redemption_amount": 0.0, "purchase_amount": 0.0}
            if _table_exists(conn, db_name, "jnl"):
                jnl_cols = _get_columns(conn, db_name, "jnl")
                jnl_date_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                jnl_line_col = _pick_column(jnl_cols, ["line", "line_code"]) or "line"
                jnl_sale_col = _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or "sale"
                jnl_price_col = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_descript_col = _pick_column(jnl_cols, ["descript", "description", "desc"]) or None

                if jnl_date_col and jnl_line_col and jnl_sale_col and jnl_price_col and jnl_descript_col:
                    sql = f"""
                        SELECT `{jnl_sale_col}` AS sale_id, `{jnl_line_col}` AS line_code,
                               `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript
                        FROM jnl
                        WHERE DATE(`{jnl_date_col}`) = :today
                        ORDER BY `{jnl_sale_col}`, `{jnl_line_col}`
                    """
                    rows = conn.execute(text(sql), {"today": today}).mappings()

                    current_sale = None
                    current_group = []
                    for row in rows:
                        sale_id = str(row.get("sale_id", ""))
                        if sale_id != current_sale:
                            if current_group:
                                lines = {r.get("line_code", "") for r in current_group}
                                has_gift = False
                                for r in current_group:
                                    if r.get("line_code") == "980":
                                        desc = str(r.get("descript", "")).lower()
                                        if any(pattern in desc for pattern in ["gift card", "giftcard", "gc", "gift cert", "giftcert", "gift certificate", "giftcertificate", "gift crd", "giftcrd"]):
                                            has_gift = True
                                            break

                                if "950" in lines and "980" in lines and has_gift:
                                    for r in current_group:
                                        if r.get("line_code") == "950":
                                            try:
                                                price_val = float(r.get("price") or 0)
                                                if price_val < 0:
                                                    gift_card_activity["redemptions"] += 1
                                                    gift_card_activity["redemption_amount"] += abs(price_val)
                                                elif price_val > 0:
                                                    gift_card_activity["purchases"] += 1
                                                    gift_card_activity["purchase_amount"] += price_val
                                            except:
                                                pass
                            current_sale = sale_id
                            current_group = [row]
                        else:
                            current_group.append(row)

                    if current_group:
                        lines = {r.get("line_code", "") for r in current_group}
                        has_gift = False
                        for r in current_group:
                            if r.get("line_code") == "980":
                                desc = str(r.get("descript", "")).lower()
                                if any(pattern in desc for pattern in ["gift card", "giftcard", "gc", "gift cert", "giftcert", "gift certificate", "giftcertificate", "gift crd", "giftcrd"]):
                                    has_gift = True
                                    break

                        if "950" in lines and "980" in lines and has_gift:
                            for r in current_group:
                                if r.get("line_code") == "950":
                                    try:
                                        price_val = float(r.get("price") or 0)
                                        if price_val < 0:
                                            gift_card_activity["redemptions"] += 1
                                            gift_card_activity["redemption_amount"] += abs(price_val)
                                        elif price_val > 0:
                                            gift_card_activity["purchases"] += 1
                                            gift_card_activity["purchase_amount"] += price_val
                                    except:
                                        pass

                    gift_card_activity["redemption_amount"] = round(gift_card_activity["redemption_amount"], 2)
                    gift_card_activity["purchase_amount"] = round(gift_card_activity["purchase_amount"], 2)

            # Recent purchase orders (last 3 days)
            recent_pos = []
            if _table_exists(conn, db_name, "poh"):
                poh_cols = _get_columns(conn, db_name, "poh")
                poh_status_col = _pick_column(poh_cols, ["status", "stat"]) or "status"
                poh_vendor_col = _pick_column(poh_cols, ["vendor", "vcode"]) or "vendor"
                poh_total_col = _pick_column(poh_cols, ["total"]) or None
                poh_rcv_col = _pick_column(poh_cols, ["rcvdate", "received_date", "rcv_date"]) or None
                poh_ord_col = _pick_column(poh_cols, ["orddate", "order_date"]) or None

                from datetime import timedelta
                three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

                where_parts = [f"poh.`{poh_status_col}` IN (4, 6, 8)"]
                date_filters = []
                if poh_rcv_col:
                    date_filters.append(f"(poh.`{poh_status_col}` IN (4, 6) AND DATE(poh.`{poh_rcv_col}`) >= :three_days_ago)")
                if poh_ord_col:
                    date_filters.append(f"(poh.`{poh_status_col}` = 8 AND DATE(poh.`{poh_ord_col}`) >= :three_days_ago)")
                if date_filters:
                    where_parts.append(f"({' OR '.join(date_filters)})")

                vnd_cols = _get_columns(conn, db_name, "vnd") if _table_exists(conn, db_name, "vnd") else []
                vnd_vendor_col = _pick_column(vnd_cols, ["vendor"])
                vnd_lastname_col = _pick_column(vnd_cols, ["lastname", "last_name", "lname"])

                select_parts = [f"poh.`{poh_vendor_col}` AS vendor_num"]
                if vnd_cols and vnd_vendor_col and vnd_lastname_col:
                    select_parts.append(f"COALESCE(vnd.`{vnd_lastname_col}`, CONCAT('Vendor ', poh.`{poh_vendor_col}`)) AS vendor_name")
                    join_part = f"LEFT JOIN vnd ON vnd.`{vnd_vendor_col}` = poh.`{poh_vendor_col}`"
                else:
                    select_parts.append(f"CONCAT('Vendor ', poh.`{poh_vendor_col}`) AS vendor_name")
                    join_part = ""

                if poh_total_col:
                    select_parts.append(f"SUM(poh.`{poh_total_col}`) AS po_total")
                else:
                    select_parts.append("0 AS po_total")

                sql = f"""
                    SELECT {', '.join(select_parts)}
                    FROM poh
                    {join_part}
                    WHERE {' AND '.join(where_parts)}
                    GROUP BY poh.`{poh_vendor_col}`
                    ORDER BY po_total DESC
                    LIMIT 10
                """
                rows = conn.execute(text(sql), {"three_days_ago": three_days_ago}).mappings()
                recent_pos = [
                    {
                        "vendor_name": str(r.get("vendor_name", "")),
                        "total": round(float(r.get("po_total", 0) or 0), 2)
                    }
                    for r in rows
                ]

            payload = {
                "store": response_meta,
                "date": today,
                "sales_summary": {
                    "total_sales": round(today_sales, 2),
                    "transaction_count": today_tx_count,
                },
                "top_items": top_items_today,
                "payment_methods": payment_methods_today,
                "gift_card_activity": gift_card_activity,
                "recent_purchase_orders": recent_pos,
            }

            freshness_info = freshness.to_payload()
            if freshness_info:
                payload["data_freshness"] = freshness_info
            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quick insights: {str(e)}")


@router.get("/product-history")
def get_product_history(
    store: Optional[str] = Query(
        default=None, description="Store identifier (store_id or store_db)"
    ),
    search: Optional[str] = Query(
        default="", description="Search by SKU, UPC, or product name"
    ),
    user: dict = Depends(get_auth_user),
):
    """Get product sales history with monthly breakdown, purchase orders, and recent sales."""
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        with engine.connect() as conn:
            response_meta = {
                "store_db": db_name,
                "store_id": selected_store.get("store_id"),
                "store_name": selected_store.get("store_name"),
            }

            products = []

            # Search for products in inv table
            if _table_exists(conn, db_name, "inv"):
                inv_cols = _get_columns(conn, db_name, "inv")
                inv_sku_col = _pick_column(inv_cols, ["sku"]) or "sku"
                inv_name_col = (
                    _pick_column(inv_cols, ["desc", "description", "name"])
                    or inv_sku_col
                )
                inv_price_col = (
                    _pick_column(inv_cols, ["price", "sprice", "sell_price"]) or None
                )
                inv_cost_col = (
                    _pick_column(inv_cols, ["lcost", "last_cost", "acost", "avg_cost"])
                    or None
                )
                inv_deleted_col = (
                    _pick_column(inv_cols, ["deleted", "del", "is_deleted"]) or None
                )

                # Get UPC from upc table if it exists
                upc_table_exists = _table_exists(conn, db_name, "upc")
                upc_sku_col = None
                upc_upc_col = None
                if upc_table_exists:
                    upc_cols = _get_columns(conn, db_name, "upc")
                    upc_sku_col = _pick_column(upc_cols, ["sku"]) or "sku"
                    upc_upc_col = _pick_column(upc_cols, ["upc", "barcode"]) or "upc"

                # Build search query
                where_parts = []
                params: Dict[str, Any] = {}

                if inv_deleted_col:
                    where_parts.append(f"UPPER(`{inv_deleted_col}`) != 'T'")

                if search:
                    search_like = f"%{search}%"
                    # Search by SKU, name, or UPC
                    search_conditions = [
                        f"`{inv_sku_col}` LIKE :search",
                        f"`{inv_name_col}` LIKE :search",
                    ]
                    if upc_table_exists and upc_upc_col:
                        # Join with upc table for UPC search
                        search_conditions.append(
                            f"EXISTS (SELECT 1 FROM upc WHERE upc.`{upc_sku_col}` = inv.`{inv_sku_col}` AND upc.`{upc_upc_col}` LIKE :search)"
                        )
                    where_parts.append(f"({' OR '.join(search_conditions)})")
                    params["search"] = search_like

                # Limit results to top 50 products for performance
                price_select = (
                    f"inv.`{inv_price_col}` AS price"
                    if inv_price_col
                    else "NULL AS price"
                )
                cost_select = (
                    f"inv.`{inv_cost_col}` AS cost" if inv_cost_col else "NULL AS cost"
                )
                upc_join = ""
                if upc_table_exists and search and upc_sku_col:
                    upc_join = (
                        f"LEFT JOIN upc ON upc.`{upc_sku_col}` = inv.`{inv_sku_col}`"
                    )

                inv_sql = f"""
                    SELECT DISTINCT inv.`{inv_sku_col}` AS sku,
                           inv.`{inv_name_col}` AS name,
                           {price_select},
                           {cost_select}
                    FROM inv
                    {upc_join}
                    WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                    ORDER BY inv.`{inv_sku_col}`
                    LIMIT 50
                """

                inv_rows = conn.execute(text(inv_sql), params).mappings()
                product_skus = []

                for inv_row in inv_rows:
                    sku = str(inv_row.get("sku", ""))
                    if not sku:
                        continue

                    product_skus.append(sku)

                    # Get UPC for this product
                    upc_value = None
                    if upc_table_exists and upc_sku_col and upc_upc_col:
                        upc_sql = f"""
                            SELECT `{upc_upc_col}` AS upc
                            FROM upc
                            WHERE `{upc_sku_col}` = :sku
                            LIMIT 1
                        """
                        upc_row = (
                            conn.execute(text(upc_sql), {"sku": sku}).mappings().first()
                        )
                        if upc_row:
                            upc_value = str(upc_row.get("upc", "")) or None

                    product = {
                        "sku": sku,
                        "upc": upc_value,
                        "name": str(inv_row.get("name", "")) or None,
                        "price": float(inv_row.get("price", 0) or 0),
                        "cost": float(inv_row.get("cost", 0) or 0),
                        "monthly_sales": [],
                        "purchase_orders": [],
                        "recent_sales": [],
                    }

                    # Get monthly sales from jnl
                    if _table_exists(conn, db_name, "jnl"):
                        jnl_cols = _get_columns(conn, db_name, "jnl")
                        jnl_sku_col = _pick_column(jnl_cols, ["sku"]) or "sku"
                        jnl_date_col = (
                            _pick_column(
                                jnl_cols,
                                [
                                    "sale_date",
                                    "trans_date",
                                    "transaction_date",
                                    "tdate",
                                    "date",
                                ],
                            )
                            or None
                        )
                        jnl_qty_col = (
                            _pick_column(jnl_cols, ["qty", "quantity"]) or None
                        )
                        jnl_price_col = (
                            _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                        )
                        jnl_rflag_col = _pick_column(jnl_cols, ["rflag"]) or None

                        if jnl_sku_col and jnl_date_col:
                            jnl_where = [f"`{jnl_sku_col}` = :sku"]
                            jnl_params = {"sku": sku}

                            if jnl_rflag_col:
                                jnl_where.append(f"`{jnl_rflag_col}` = 0")

                            # Group by year-month for monthly totals
                            qty_expr = f"`{jnl_qty_col}`" if jnl_qty_col else "1"
                            price_expr = f"`{jnl_price_col}`" if jnl_price_col else "0"

                            monthly_sql = f"""
                                SELECT 
                                    DATE_FORMAT(DATE(`{jnl_date_col}`), '%Y-%m') AS month,
                                    SUM({qty_expr}) AS total_qty,
                                    SUM({price_expr}) AS total_amount
                                FROM jnl
                                WHERE {' AND '.join(jnl_where)}
                                GROUP BY DATE_FORMAT(DATE(`{jnl_date_col}`), '%Y-%m')
                                ORDER BY month DESC
                                LIMIT 24
                            """

                            monthly_rows = conn.execute(
                                text(monthly_sql), jnl_params
                            ).mappings()
                            for month_row in monthly_rows:
                                product["monthly_sales"].append(
                                    {
                                        "month": str(month_row.get("month", "")),
                                        "total_qty": float(
                                            month_row.get("total_qty", 0) or 0
                                        ),
                                        "total_amount": float(
                                            month_row.get("total_amount", 0) or 0
                                        ),
                                    }
                                )

                    # Get last 5 purchase orders from pod
                    if _table_exists(conn, db_name, "pod"):
                        pod_cols = _get_columns(conn, db_name, "pod")
                        pod_sku_col = _pick_column(pod_cols, ["sku"]) or "sku"
                        pod_po_col = (
                            _pick_column(pod_cols, ["po", "po_id", "poh_id", "invno"])
                            or None
                        )
                        pod_qty_col = (
                            _pick_column(pod_cols, ["qty", "quantity"]) or None
                        )
                        pod_price_col = (
                            _pick_column(pod_cols, ["price", "amount", "total"]) or None
                        )
                        pod_date_col = (
                            _pick_column(pod_cols, ["date", "pdate", "created_at"])
                            or None
                        )

                        if pod_sku_col and pod_po_col:
                            pod_select_parts = [f"`{pod_po_col}` AS po_id"]
                            pod_group_parts = [f"`{pod_po_col}`"]

                            if pod_date_col:
                                pod_select_parts.append(
                                    f"DATE(`{pod_date_col}`) AS po_date"
                                )
                                pod_group_parts.append(f"DATE(`{pod_date_col}`)")
                            if pod_qty_col:
                                pod_select_parts.append(f"SUM(`{pod_qty_col}`) AS qty")
                            if pod_price_col:
                                pod_select_parts.append(
                                    f"SUM(`{pod_price_col}`) AS amount"
                                )

                            pod_sql = f"""
                                SELECT {', '.join(pod_select_parts)}
                                FROM pod
                                WHERE `{pod_sku_col}` = :sku
                                GROUP BY {', '.join(pod_group_parts)}
                                ORDER BY `{pod_po_col}` DESC
                                LIMIT 5
                            """

                            pod_rows = conn.execute(
                                text(pod_sql), {"sku": sku}
                            ).mappings()
                            for pod_row in pod_rows:
                                product["purchase_orders"].append(
                                    {
                                        "po_id": str(pod_row.get("po_id", "")),
                                        "date": (
                                            str(pod_row.get("po_date", ""))
                                            if pod_date_col
                                            else None
                                        ),
                                        "qty": (
                                            float(pod_row.get("qty", 0) or 0)
                                            if pod_qty_col
                                            else 0
                                        ),
                                        "amount": (
                                            float(pod_row.get("amount", 0) or 0)
                                            if pod_price_col
                                            else 0
                                        ),
                                    }
                                )

                    # Get last 5 sales from jnl
                    if _table_exists(conn, db_name, "jnl"):
                        jnl_cols = _get_columns(conn, db_name, "jnl")
                        jnl_sku_col = _pick_column(jnl_cols, ["sku"]) or "sku"
                        jnl_sale_col = (
                            _pick_column(jnl_cols, ["sale", "sale_id", "invno"])
                            or "sale"
                        )
                        jnl_date_col = (
                            _pick_column(
                                jnl_cols,
                                [
                                    "sale_date",
                                    "trans_date",
                                    "transaction_date",
                                    "tdate",
                                    "date",
                                ],
                            )
                            or None
                        )
                        jnl_qty_col = (
                            _pick_column(jnl_cols, ["qty", "quantity"]) or None
                        )
                        jnl_price_col = (
                            _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                        )
                        jnl_rflag_col = _pick_column(jnl_cols, ["rflag"]) or None

                        if jnl_sku_col and jnl_sale_col and jnl_date_col:
                            jnl_where = [f"`{jnl_sku_col}` = :sku"]
                            jnl_params = {"sku": sku}

                            if jnl_rflag_col:
                                jnl_where.append(f"`{jnl_rflag_col}` = 0")

                            sales_select_parts = [
                                f"`{jnl_sale_col}` AS sale_id",
                                f"DATE(`{jnl_date_col}`) AS sale_date",
                            ]
                            if jnl_qty_col:
                                sales_select_parts.append(
                                    f"SUM(`{jnl_qty_col}`) AS qty"
                                )
                            if jnl_price_col:
                                sales_select_parts.append(
                                    f"SUM(`{jnl_price_col}`) AS amount"
                                )

                            sales_sql = f"""
                                SELECT {', '.join(sales_select_parts)}
                                FROM jnl
                                WHERE {' AND '.join(jnl_where)}
                                GROUP BY `{jnl_sale_col}`, DATE(`{jnl_date_col}`)
                                ORDER BY `{jnl_date_col}` DESC, `{jnl_sale_col}` DESC
                                LIMIT 5
                            """

                            sales_rows = conn.execute(
                                text(sales_sql), jnl_params
                            ).mappings()
                            for sale_row in sales_rows:
                                product["recent_sales"].append(
                                    {
                                        "sale_id": str(sale_row.get("sale_id", "")),
                                        "date": (
                                            str(sale_row.get("sale_date", ""))
                                            if sale_row.get("sale_date")
                                            else None
                                        ),
                                        "qty": (
                                            float(sale_row.get("qty", 0) or 0)
                                            if jnl_qty_col
                                            else 0
                                        ),
                                        "amount": (
                                            float(sale_row.get("amount", 0) or 0)
                                            if jnl_price_col
                                            else 0
                                        ),
                                    }
                                )

                    products.append(product)

            return {
                "products": products,
                "meta": response_meta,
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get product history: {str(e)}"
        )


@router.get("/diagnostics")
def get_diagnostics(
    store: Optional[str] = Query(
        default=None, description="Store identifier (store_id or store_db)"
    ),
    user: dict = Depends(get_auth_user),
):
    """
    Diagnostic endpoint to check database performance and identify slow queries.
    Shows active processes, slow queries, and connection pool status.
    """
    try:
        selected_store = _select_store(user, store)
        db_name = selected_store["store_db"]
        db_user = selected_store["db_user"]
        db_pass = selected_store["db_pass"]

        engine = _get_engine(db_user, db_pass, db_name)

        diagnostics = {
            "database": db_name,
            "connection_pool": {
                "pool_size": _POOL_SIZE,
                "max_overflow": _POOL_MAX_OVERFLOW,
                "pool_timeout": _POOL_TIMEOUT,
                "query_timeout": _QUERY_TIMEOUT,
            },
            "active_queries": [],
            "slow_queries": [],
            "process_list": [],
        }

        with engine.connect() as conn:
            # Get active processes/queries
            try:
                process_list = (
                    conn.execute(
                        text(
                            """
                    SELECT 
                        ID,
                        USER,
                        HOST,
                        DB,
                        COMMAND,
                        TIME,
                        STATE,
                        LEFT(INFO, 200) AS QUERY
                    FROM information_schema.PROCESSLIST
                    WHERE DB = :db_name AND COMMAND != 'Sleep'
                    ORDER BY TIME DESC
                    LIMIT 20
                """
                        ),
                        {"db_name": db_name},
                    )
                    .mappings()
                    .all()
                )

                diagnostics["process_list"] = [
                    {
                        "id": int(p.get("ID", 0)),
                        "user": str(p.get("USER", "")),
                        "host": str(p.get("HOST", "")),
                        "command": str(p.get("COMMAND", "")),
                        "time": int(p.get("TIME", 0)),
                        "state": str(p.get("STATE", "")),
                        "query": str(p.get("QUERY", ""))[:200],
                    }
                    for p in process_list
                ]
            except Exception as e:
                diagnostics["process_list_error"] = str(e)

            # Get slow query log (if enabled)
            try:
                slow_queries = (
                    conn.execute(
                        text(
                            """
                    SELECT 
                        start_time,
                        user_host,
                        query_time,
                        lock_time,
                        rows_sent,
                        rows_examined,
                        LEFT(sql_text, 500) AS sql_text
                    FROM mysql.slow_log
                    WHERE start_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
                    ORDER BY start_time DESC
                    LIMIT 10
                """
                        )
                    )
                    .mappings()
                    .all()
                )

                diagnostics["slow_queries"] = [
                    {
                        "start_time": str(s.get("start_time", "")),
                        "query_time": float(s.get("query_time", 0)),
                        "lock_time": float(s.get("lock_time", 0)),
                        "rows_examined": int(s.get("rows_examined", 0)),
                        "sql_text": str(s.get("sql_text", ""))[:500],
                    }
                    for s in slow_queries
                ]
            except Exception as e:
                diagnostics["slow_log_error"] = str(e)
                diagnostics["slow_log_note"] = "Slow query log may not be enabled"

            # Get connection pool status
            try:
                pool_status = (
                    conn.execute(
                        text(
                            """
                    SELECT 
                        VARIABLE_NAME,
                        VARIABLE_VALUE
                    FROM information_schema.GLOBAL_STATUS
                    WHERE VARIABLE_NAME IN (
                        'Threads_connected',
                        'Threads_running',
                        'Max_used_connections',
                        'Questions',
                        'Slow_queries'
                    )
                """
                        )
                    )
                    .mappings()
                    .all()
                )

                diagnostics["database_status"] = {
                    str(p.get("VARIABLE_NAME", "")): str(p.get("VARIABLE_VALUE", ""))
                    for p in pool_status
                }
            except Exception as e:
                diagnostics["status_error"] = str(e)

        return diagnostics

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get diagnostics: {str(e)}"
        )
