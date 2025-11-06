from fastapi import APIRouter, Depends, HTTPException, Query
from app.auth import get_auth_user
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import OrderedDict
import os
import atexit

"""
This module re-exports the sales insights router that dynamically adapts to
Visual FoxPro schema differences (for example, DESCRIPTION vs DESCRIPT). The
main application continues to import the router from ``app.insights`` so we keep
that import path stable.
"""

router = APIRouter()

# Engine cache to reuse engines per database connection string
_engine_cache: "OrderedDict[str, Any]" = OrderedDict()

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


# Cleanup function to dispose of all engines on shutdown
def _dispose_engines():
    for engine in _engine_cache.values():
        try:
            engine.dispose()
        except Exception:
            pass
    _engine_cache.clear()


atexit.register(_dispose_engines)

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
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
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

            # Optional filters to exclude non-item or return rows if present
            where_clauses = []
            if detected.get("has_rflag"):
                where_clauses.append("`rflag` = 0")
            if detected.get("has_sku"):
                where_clauses.append("`sku` > 0")
            # Date range filters on DATE(date_col)
            params = {}
            if start:
                where_clauses.append("DATE(`{}`) >= :start".format(date_col))
                params["start"] = start
            if end:
                where_clauses.append("DATE(`{}`) <= :end".format(date_col))
                params["end"] = end
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sql = f"""
                SELECT 
                    DATE(`{date_col}`) AS date,
                    {qty_expr} AS total_items_sold,
                    {amount_expr} AS total_sales
                FROM `{table}`
                {where_sql}
                GROUP BY DATE(`{date_col}`)
                ORDER BY DATE(`{date_col}`) DESC
                LIMIT 7
            """

            result = conn.execute(text(sql), params).mappings()

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

            # Build summary
            gross_total = sum(row["total_sales"] for row in sales_data)
            total_items = sum(row["total_items_sold"] for row in sales_data)
            days_captured = len(sales_data)
            summary = {
                "gross_sales": round(gross_total, 2),
                "total_items": float(total_items),
                "days_captured": days_captured,
                "average_daily_sales": round(gross_total / days_captured, 2) if days_captured else 0.0,
            }

            # Hourly breakdown: prefer jnh.tstamp joined by jnl sale id, but filter dates using jnl's date column
            hourly: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnh"):
                jnh_cols = _get_columns(conn, db_name, "jnh")
                jnh_time = _pick_column(jnh_cols, ["tstamp", "timestamp", "time", "t_time"]) or "tstamp"
                jnh_sale = _pick_column(jnh_cols, ["sale", "sale_id", "invno"]) or "sale"

                jnl_cols = _get_columns(conn, db_name, "jnl") if _table_exists(conn, db_name, "jnl") else []
                jnl_sale = _pick_column(jnl_cols, ["sale", "sale_id", "invno"]) or jnh_sale
                jnl_rflag = "rflag" if "rflag" in jnl_cols else None
                jnl_sku = "sku" if "sku" in jnl_cols else None
                jnl_qty = _pick_column(jnl_cols, ["qty", "quantity"]) or None
                jnl_price = _pick_column(jnl_cols, ["amount", "price", "total"]) or None
                jnl_date_filter_col = _pick_column(jnl_cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None

                where_parts = ["1=1"]
                params2: Dict[str, Any] = {}
                # Filter by date using jnl's date column (jnh.tstamp is time-only on many schemas)
                if jnl_date_filter_col and start:
                    where_parts.append(f"DATE(jnl.`{jnl_date_filter_col}`) >= :h_start")
                    params2["h_start"] = start
                if jnl_date_filter_col and end:
                    where_parts.append(f"DATE(jnl.`{jnl_date_filter_col}`) <= :h_end")
                    params2["h_end"] = end
                if jnl_rflag:
                    where_parts.append(f"jnl.`{jnl_rflag}` = 0")
                if jnl_sku:
                    where_parts.append(f"jnl.`{jnl_sku}` > 0")

                amount_expr_h = None
                if jnl_price and jnl_qty:
                    amount_expr_h = f"SUM(jnl.`{jnl_qty}` * jnl.`{jnl_price}`)"
                elif jnl_price:
                    amount_expr_h = f"SUM(jnl.`{jnl_price}`)"
                else:
                    amount_expr_h = "SUM(0)"

                hourly_sql = f"""
                    SELECT HOUR(jnh.`{jnh_time}`) AS hour, {amount_expr_h} AS total_sales
                    FROM jnh
                    JOIN jnl ON jnh.`{jnh_sale}` = jnl.`{jnl_sale}`
                    WHERE {' AND '.join(where_parts)}
                    GROUP BY HOUR(jnh.`{jnh_time}`)
                    ORDER BY hour
                """
                rows = conn.execute(text(hourly_sql), params2).mappings()
                hourly = [{"hour": f"{int(r['hour']):02d}:00", "total_sales": float(r["total_sales"]) } for r in rows]

            # Payment methods from jnl tenders 980-989 (filter dates using jnl's date column)
            payment_methods: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl"):
                cols = _get_columns(conn, db_name, "jnl")
                cat_col = "cat" if "cat" in cols else None
                jnl_date_filter_col = _pick_column(cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                amt_col = None
                for c in ["amount", "total", "price"]:
                    if c in cols:
                        amt_col = c
                        break
                where_parts = ["1=1"]
                params3: Dict[str, Any] = {}
                if jnl_date_filter_col and start:
                    where_parts.append(f"DATE(`{jnl_date_filter_col}`) >= :p_start")
                    params3["p_start"] = start
                if jnl_date_filter_col and end:
                    where_parts.append(f"DATE(`{jnl_date_filter_col}`) <= :p_end")
                    params3["p_end"] = end
                if cat_col and amt_col:
                    tender_sql = f"""
                        SELECT `{cat_col}` AS code, SUM(`{amt_col}`) AS total
                        FROM jnl
                        WHERE {' AND '.join(where_parts)} AND `{cat_col}` BETWEEN 980 AND 989
                        GROUP BY `{cat_col}`
                    """
                    rows = conn.execute(text(tender_sql), params3).mappings()
                    label_map = {
                        980: "Cash", 981: "Check", 982: "Visa", 983: "MasterCard",
                        984: "AmEx", 985: "Discover", 986: "Debit", 987: "Gift Card",
                        988: "House", 989: "EBT",
                    }
                    total_sum = 0.0
                    tmp = []
                    for r in rows:
                        code = int(r["code"]) if r["code"] is not None else None
                        amount = float(r["total"] or 0)
                        total_sum += amount
                        tmp.append({"method": label_map.get(code, str(code)), "total_sales": amount})
                    # add percentages
                    if total_sum:
                        payment_methods = [
                            {**item, "percentage": round((item["total_sales"]/total_sum)*100, 2)} for item in tmp
                        ]

            # Categories: jnl.cat like 'P%' joined to cat table if present
            categories: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl"):
                cols = _get_columns(conn, db_name, "jnl")
                if "cat" in cols:
                    # Determine jnl date/qty/price columns
                    jnl_date_col = _pick_column(cols, ["sale_date", "trans_date", "transaction_date", "tdate", "date"]) or None
                    jnl_qty_col = _pick_column(cols, ["qty", "quantity"]) or None
                    jnl_price_col = _pick_column(cols, ["amount", "total", "price"]) or None

                    where_parts = ["1=1", "jnl.`sku` > 0", "jnl.`rflag` = 0", "jnl.`cat` LIKE 'P%'"]
                    params4: Dict[str, Any] = {}
                    if start and jnl_date_col:
                        where_parts.append(f"DATE(jnl.`{jnl_date_col}`) >= :c_start")
                        params4["c_start"] = start
                    if end and jnl_date_col:
                        where_parts.append(f"DATE(jnl.`{jnl_date_col}`) <= :c_end")
                        params4["c_end"] = end

                    # Compute amount expression specific to jnl
                    if jnl_price_col and jnl_qty_col:
                        amt_expr = f"SUM(jnl.`{jnl_qty_col}` * jnl.`{jnl_price_col}`)"
                    elif jnl_price_col:
                        amt_expr = f"SUM(jnl.`{jnl_price_col}`)"
                    else:
                        amt_expr = "SUM(0)"

                    cat_join_label = None
                    if _table_exists(conn, db_name, "cat"):
                        cat_cols = _get_columns(conn, db_name, "cat")
                        cat_code = _pick_column(cat_cols, ["cat", "code", "id"]) or "cat"
                        cat_label = _pick_column(cat_cols, ["desc", "description", "name", "label"]) or cat_code
                        cat_join_label = (cat_code, cat_label)
                    if cat_join_label:
                        cc, cl = cat_join_label
                        cat_sql = f"""
                            SELECT COALESCE(cat.`{cl}`, jnl.`cat`) AS category, {amt_expr} AS total
                            FROM jnl LEFT JOIN cat ON cat.`{cc}` = jnl.`cat`
                            WHERE {' AND '.join(where_parts)}
                            GROUP BY COALESCE(cat.`{cl}`, jnl.`cat`)
                            ORDER BY total DESC
                        """
                    else:
                        cat_sql = f"""
                            SELECT jnl.`cat` AS category, {amt_expr} AS total
                            FROM jnl
                            WHERE {' AND '.join(where_parts)}
                            GROUP BY jnl.`cat`
                            ORDER BY total DESC
                        """
                    rows = conn.execute(text(cat_sql), params4).mappings()
                    categories = [{"category": str(r["category"]), "total_sales": float(r["total"]) } for r in rows]

            # Top products from hst by qty
            top_items: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "hst"):
                hst_cols = _get_columns(conn, db_name, "hst")
                hst_date = _pick_column(hst_cols, ["date", "tdate", "sale_date", "trans_date"]) or "date"
                hst_qty = _pick_column(hst_cols, ["qty", "quantity"]) or "qty"
                hst_sku = _pick_column(hst_cols, ["sku"]) or "sku"
                where_parts = ["1=1"]
                params5: Dict[str, Any] = {}
                if start:
                    where_parts.append(f"DATE(`{hst_date}`) >= :t_start")
                    params5["t_start"] = start
                if end:
                    where_parts.append(f"DATE(`{hst_date}`) <= :t_end")
                    params5["t_end"] = end
                base_sql = f"""
                    SELECT `{hst_sku}` AS sku, SUM(`{hst_qty}`) AS total_qty
                    FROM hst
                    WHERE {' AND '.join(where_parts)} AND `{hst_sku}` > 0
                    GROUP BY `{hst_sku}`
                    ORDER BY total_qty DESC
                    LIMIT 20
                """
                rows = conn.execute(text(base_sql), params5).mappings()
                # Join descriptions from inv
                inv_names: Dict[str, str] = {}
                skus = [str(r["sku"]) for r in rows]
                if skus and _table_exists(conn, db_name, "inv"):
                    inv_cols = _get_columns(conn, db_name, "inv")
                    inv_sku = _pick_column(inv_cols, ["sku"]) or "sku"
                    inv_name = _pick_column(inv_cols, ["desc", "description", "name"]) or inv_sku
                    in_params = {"skus": tuple(skus)}
                    inv_sql = text(f"SELECT `{inv_sku}` AS sku, `{inv_name}` AS name FROM inv WHERE `{inv_sku}` IN :skus").bindparams()
                    inv_rows = conn.execute(inv_sql, in_params).mappings()
                    for r in inv_rows:
                        inv_names[str(r["sku"])]= str(r["name"]) if r["name"] is not None else ""
                top_items = [
                    {
                        "sku": str(r["sku"]),
                        "description": inv_names.get(str(r["sku"]), ""),
                        "total_items_sold": float(r["total_qty"]),
                        "total_sales": 0.0,
                    }
                    for r in rows
                ]

            # Purchase orders: today's POs grouped by vendor, with expandable order details from pod
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

                # Use today's date from the selected date range (or default to today)
                today_str = end if end else datetime.now().strftime('%Y-%m-%d')

                # Group by vendor for today's POs
                where_parts = [f"poh.`{poh_status_col}` IN (4, 6, 8)"]
                params: Dict[str, Any] = {}

                # Date filters: status 4,6 use rcvdate; status 8 uses orddate - filter to today only
                date_filters = []
                if poh_rcv_col:
                    date_filters.append(f"(poh.`{poh_status_col}` IN (4, 6) AND DATE(poh.`{poh_rcv_col}`) = :today)")
                if poh_ord_col:
                    date_filters.append(f"(poh.`{poh_status_col}` = 8 AND DATE(poh.`{poh_ord_col}`) = :today)")
                if date_filters:
                    where_parts.append(f"({' OR '.join(date_filters)})")
                    params["today"] = today_str

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
                status_map = {4: "posted", 6: "received", 8: "open"}
                
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
                        
                        # Get POs from poh for this vendor with today's date and matching status
                        poh_ids_where = [f"poh.`{poh_vendor_col}` = :vnum", f"poh.`{poh_status_col}` = :stat"]
                        poh_ids_params = {"vnum": vendor_num, "stat": status_code}
                        
                        if date_filters:
                            poh_ids_where.append(f"({' OR '.join(date_filters)})")
                            poh_ids_params.update(params)
                        
                        poh_ids_sql = f"""
                            SELECT poh.`{poh_id_col}` AS po_id
                            FROM poh
                            WHERE {' AND '.join(poh_ids_where)}
                        """
                        poh_id_rows = conn.execute(text(poh_ids_sql), poh_ids_params).mappings()
                        po_ids = [str(row.get("po_id")) for row in poh_id_rows if row.get("po_id")]
                        
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
                            
                            pod_sql = f"""
                                SELECT {', '.join(pod_select)}
                                FROM pod
                                WHERE {' AND '.join(pod_where)}
                                GROUP BY `{pod_po_col}`
                                ORDER BY `{pod_po_col}` DESC
                            """
                            pod_rows = conn.execute(text(pod_sql), pod_params).mappings()
                            for pod_row in pod_rows:
                                order_details.append({
                                    "po_id": str(pod_row.get("po_id", "")),
                                    "order_total": float(pod_row.get("order_total", 0) or 0),
                                    "order_date": str(pod_row.get("order_date", "")) if pod_row.get("order_date") else None,
                                })
                    
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

            return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sales insights: {str(e)}")


@router.get("/operations")
def get_operations_insights(
    store: Optional[str] = Query(default=None, description="Store identifier (store_id or store_db)"),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    user: dict = Depends(get_auth_user),
):
    """Get store operations insights: transaction counts, gift cards, products, supplier invoices."""
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

            # Default date range to last 30 days if not provided
            if not start:
                from datetime import timedelta
                start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')

            # Transaction counts and till amounts from jnl
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

                if jnl_date_col and jnl_line_col and jnl_sale_col:
                    # Get all transactions in date range
                    where_parts = []
                    params: Dict[str, Any] = {}
                    if start:
                        where_parts.append(f"DATE(`{jnl_date_col}`) >= :start")
                        params["start"] = start
                    if end:
                        where_parts.append(f"DATE(`{jnl_date_col}`) <= :end")
                        params["end"] = end
                    if jnl_rflag_col:
                        where_parts.append(f"`{jnl_rflag_col}` = 0")

                    # Get all rows for grouping by sale
                    sql = f"""
                        SELECT `{jnl_sale_col}` AS sale_id, `{jnl_line_col}` AS line_code,
                               `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript
                        FROM jnl
                        WHERE {' AND '.join(where_parts) if where_parts else '1=1'}
                        ORDER BY `{jnl_sale_col}`, `{jnl_line_col}`
                    """
                    rows = conn.execute(text(sql), params).mappings()
                    
                    # Group by sale_id to find valid transactions
                    current_sale = None
                    current_group = []
                    for row in rows:
                        sale_id = str(row.get("sale_id", ""))
                        line_code = str(row.get("line_code", ""))
                        
                        if sale_id != current_sale:
                            # Process previous group
                            if current_group:
                                lines = {r.get("line_code", "") for r in current_group}
                                if "950" in lines and "980" in lines:
                                    tx_count += 1
                                    # Sum till from line 950
                                    for r in current_group:
                                        if r.get("line_code") == "950":
                                            try:
                                                price_val = float(r.get("price") or 0)
                                                if abs(price_val) <= 100000:
                                                    total_till += price_val
                                            except:
                                                pass
                                    
                                    # Check for gift card patterns in line 980 (tender line)
                                    if jnl_descript_col:
                                        has_gift_pattern = False
                                        for r in current_group:
                                            if r.get("line_code") == "980":
                                                desc = str(r.get("descript", "")).lower()
                                                if any(pattern in desc for pattern in ["gift card", "giftcard", "gc", "gift cert", "giftcert", "gift certificate", "giftcertificate", "gift crd", "giftcrd"]):
                                                    has_gift_pattern = True
                                                    break
                                        
                                        if has_gift_pattern:
                                            # Check if it's a redemption (negative price) or purchase (positive)
                                            for r in current_group:
                                                if r.get("line_code") == "950":
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
                        lines = {r.get("line_code", "") for r in current_group}
                        if "950" in lines and "980" in lines:
                            tx_count += 1
                            for r in current_group:
                                if r.get("line_code") == "950":
                                    try:
                                        price_val = float(r.get("price") or 0)
                                        if abs(price_val) <= 100000:
                                            total_till += price_val
                                    except:
                                        pass

            # Product counts from inv
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

            # Supplier invoice counts from poh
            supplier_invoice_count = 0
            first_supplier_date = None
            last_supplier_date = None
            if _table_exists(conn, db_name, "poh"):
                poh_cols = _get_columns(conn, db_name, "poh")
                poh_status_col = _pick_column(poh_cols, ["status", "stat"]) or "status"
                poh_vendor_col = _pick_column(poh_cols, ["vendor", "vcode"]) or "vendor"
                poh_rcvdate_col = _pick_column(poh_cols, ["rcvdate", "received_date", "rcv_date"]) or None
                
                where_parts = [
                    f"`{poh_status_col}` IN ('3', '4', 3, 4)",
                    f"`{poh_vendor_col}` NOT IN ('9998', '9999', 9998, 9999)"
                ]
                
                if poh_rcvdate_col:
                    if start:
                        where_parts.append(f"DATE(`{poh_rcvdate_col}`) >= :poh_start")
                    if end:
                        where_parts.append(f"DATE(`{poh_rcvdate_col}`) <= :poh_end")
                    
                    params_poh: Dict[str, Any] = {}
                    if start:
                        params_poh["poh_start"] = start
                    if end:
                        params_poh["poh_end"] = end
                    
                    sql = f"""
                        SELECT COUNT(*) AS cnt,
                               MIN(DATE(`{poh_rcvdate_col}`)) AS first_date,
                               MAX(DATE(`{poh_rcvdate_col}`)) AS last_date
                        FROM poh
                        WHERE {' AND '.join(where_parts)}
                    """
                    r = conn.execute(text(sql), params_poh).mappings().first()
                    if r:
                        supplier_invoice_count = int(r.get("cnt", 0) or 0)
                        first_supplier_date = str(r.get("first_date", "")) if r.get("first_date") else None
                        last_supplier_date = str(r.get("last_date", "")) if r.get("last_date") else None
                else:
                    sql = f"""
                        SELECT COUNT(*) AS cnt
                        FROM poh
                        WHERE {' AND '.join(where_parts)}
                    """
                    r = conn.execute(text(sql)).mappings().first()
                    if r:
                        supplier_invoice_count = int(r.get("cnt", 0) or 0)

            return {
                "store": response_meta,
                "transaction_count": tx_count,
                "total_till": round(total_till, 2),
                "gift_cards": {
                    "redemptions": {
                        "count": gift_redeem_count,
                        "amount": round(gift_redeem_amount, 2)
                    },
                    "purchases": {
                        "count": gift_purchase_count,
                        "amount": round(gift_purchase_amount, 2)
                    }
                },
                "products": {
                    "count": product_count,
                    "first_created": first_product_date,
                    "last_created": last_product_date
                },
                "supplier_invoices": {
                    "count": supplier_invoice_count,
                    "first_date": first_supplier_date,
                    "last_date": last_supplier_date
                }
            }

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
                        SELECT DATE(`{jnl_date_col}`) AS date, `{jnl_price_col}` AS price,
                               `{jnl_descript_col}` AS descript
                        FROM jnl
                        WHERE {' AND '.join(where_parts)}
                        ORDER BY DATE(`{jnl_date_col}`), `{jnl_descript_col}`
                    """
                    rows = conn.execute(text(sql), params).mappings()
                    
                    # Group by date and payment type
                    from collections import defaultdict
                    by_type = defaultdict(lambda: {"sale_amount": 0.0, "sale_count": 0, "reversal_amount": 0.0, "reversal_count": 0})
                    by_date_type = defaultdict(lambda: {"sale_amount": 0.0, "sale_count": 0, "reversal_amount": 0.0, "reversal_count": 0})
                    
                    for row in rows:
                        try:
                            price = float(row.get("price") or 0)
                            descript = str(row.get("descript", "") or "")
                            date_str = str(row.get("date", "")) if row.get("date") else ""
                            
                            payment_type = _normalize_payment_type(descript)
                            
                            if price >= 0:
                                by_type[payment_type]["sale_amount"] += price
                                by_type[payment_type]["sale_count"] += 1
                                if date_str:
                                    by_date_type[(date_str, payment_type)]["sale_amount"] += price
                                    by_date_type[(date_str, payment_type)]["sale_count"] += 1
                            else:
                                by_type[payment_type]["reversal_amount"] += abs(price)
                                by_type[payment_type]["reversal_count"] += 1
                                if date_str:
                                    by_date_type[(date_str, payment_type)]["reversal_amount"] += abs(price)
                                    by_date_type[(date_str, payment_type)]["reversal_count"] += 1
                        except:
                            continue
                    
                    # Build tenders summary
                    for payment_type, stats in by_type.items():
                        tenders_data.append({
                            "payment_type": payment_type,
                            "sale_amount": round(stats["sale_amount"], 2),
                            "sale_count": stats["sale_count"],
                            "reversal_amount": round(stats["reversal_amount"], 2),
                            "reversal_count": stats["reversal_count"],
                            "net_amount": round(stats["sale_amount"] - stats["reversal_amount"], 2)
                        })
                    
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

            return {
                "store": response_meta,
                "tenders": sorted(tenders_data, key=lambda x: x["net_amount"], reverse=True),
                "daily_trends": daily_trends
            }

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
                        SELECT `{jnl_price_col}` AS price, `{jnl_descript_col}` AS descript
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

            return {
                "store": response_meta,
                "credit_cards": sorted(credit_card_data, key=lambda x: x["total_sales"], reverse=True),
                "summary": {
                    "total_credit_card_sales": round(total_cc_sales, 2),
                    "total_ecommerce_sales": round(total_ecom_sales, 2),
                    "total_combined": round(total_cc_sales + total_ecom_sales, 2)
                }
            }

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
                hst_sku = _pick_column(hst_cols, ["sku"]) or "sku"
                
                sql = f"""
                    SELECT `{hst_sku}` AS sku, SUM(`{hst_qty}`) AS total_qty
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

            return {
                "store": response_meta,
                "date": today,
                "sales_summary": {
                    "total_sales": round(today_sales, 2),
                    "transaction_count": today_tx_count
                },
                "top_items": top_items_today,
                "payment_methods": payment_methods_today,
                "gift_card_activity": gift_card_activity,
                "recent_purchase_orders": recent_pos
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quick insights: {str(e)}")
