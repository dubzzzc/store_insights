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

        # Create engine dynamically per user
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com/{db_name}")

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

            # Hourly breakdown: prefer jnh.tstamp joined by sale id
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

                where_parts = ["1=1"]
                params2: Dict[str, Any] = {}
                if start:
                    where_parts.append(f"DATE(jnh.`{jnh_time}`) >= :h_start")
                    params2["h_start"] = start
                if end:
                    where_parts.append(f"DATE(jnh.`{jnh_time}`) <= :h_end")
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

            # Payment methods from jnl tenders 980-989
            payment_methods: List[Dict[str, Any]] = []
            if _table_exists(conn, db_name, "jnl"):
                cols = _get_columns(conn, db_name, "jnl")
                cat_col = "cat" if "cat" in cols else None
                date_expr = f"DATE(`{date_col}`)" if table == "jnl" else f"DATE(`{date_col}`)"
                amt_col = None
                for c in ["amount", "total", "price"]:
                    if c in cols:
                        amt_col = c
                        break
                where_parts = ["1=1"]
                params3: Dict[str, Any] = {}
                if start:
                    where_parts.append(f"{date_expr} >= :p_start")
                    params3["p_start"] = start
                if end:
                    where_parts.append(f"{date_expr} <= :p_end")
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

            # Purchase orders: count by status with date filters
            # Status 4 and 6 filter on rcvdate; status 8 filters on orddate. Sum counts across poh/pod if both exist.
            purchase_orders: List[Dict[str, Any]] = []
            def _po_counts_for_table(table: str) -> Dict[int, int]:
                if not _table_exists(conn, db_name, table):
                    return {}
                cols = _get_columns(conn, db_name, table)
                status_col = _pick_column(cols, ["status", "stat"]) or "status"
                rcv_col = _pick_column(cols, ["rcvdate", "received_date", "rcv_date"]) or None
                ord_col = _pick_column(cols, ["orddate", "order_date"]) or None
                counts: Dict[int, int] = {}
                # Helper to run count with optional date column
                def run_count(status_val: int, date_col: Optional[str]) -> int:
                    where_parts = [f"`{status_col}` = :s"]
                    params: Dict[str, Any] = {"s": status_val}
                    if date_col and start:
                        where_parts.append(f"DATE(`{date_col}`) >= :ds")
                        params["ds"] = start
                    if date_col and end:
                        where_parts.append(f"DATE(`{date_col}`) <= :de")
                        params["de"] = end
                    sql = f"SELECT COUNT(*) AS c FROM {table} WHERE {' AND '.join(where_parts)}"
                    row = conn.execute(text(sql), params).mappings().first()
                    return int((row or {}).get("c", 0) or 0)
                # 4=posted, 6=received use rcvdate
                counts[4] = run_count(4, rcv_col)
                counts[6] = run_count(6, rcv_col)
                # 8=open uses orddate
                counts[8] = run_count(8, ord_col)
                return counts

            total_counts: Dict[int, int] = {4: 0, 6: 0, 8: 0}
            for table in ("poh", "pod"):
                for k, v in _po_counts_for_table(table).items():
                    total_counts[k] = total_counts.get(k, 0) + v

            status_map = {4: "posted", 6: "received", 8: "open"}
            for code in (4, 6, 8):
                purchase_orders.append({
                    "status": status_map[code],
                    "order_count": int(total_counts.get(code, 0)),
                })

            # Inventory value via inv lcost/acost * onhand
            inventory = None
            if _table_exists(conn, db_name, "inv"):
                inv_cols = _get_columns(conn, db_name, "inv")
                qty_col = _pick_column(inv_cols, ["onhand", "qty", "stock"]) or None
                lcost_col = _pick_column(inv_cols, ["lcost", "last_cost"]) or None
                acost_col = _pick_column(inv_cols, ["acost", "avg_cost"]) or None
                if qty_col and (lcost_col or acost_col):
                    parts = []
                    if lcost_col:
                        parts.append(f"SUM(COALESCE(`{qty_col}`,0) * COALESCE(`{lcost_col}`,0)) AS total_lcost")
                    if acost_col:
                        parts.append(f"SUM(COALESCE(`{qty_col}`,0) * COALESCE(`{acost_col}`,0)) AS total_acost")
                    inv_sql = f"SELECT {', '.join(parts)} FROM inv"
                    r = conn.execute(text(inv_sql)).mappings().first()
                    inventory = {
                        "total_value": float((r or {}).get("total_lcost", 0) or 0),
                        "segments": [],
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
