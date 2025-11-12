-- Indexes to optimize hourly sales query performance
-- Run these on your database to improve query speed
-- Note: If an index already exists, you'll get an error - that's okay, just skip that one

-- Most important: Composite index on jnh for date filtering and sale joins
CREATE INDEX idx_jnh_tstamp_sale ON jnh(tstamp, sale);

-- Composite index on jnl for tender checks (sale, line, rflag)
CREATE INDEX idx_jnl_sale_line_rflag ON jnl(sale, line, rflag);

-- Composite index on jnl for void sales checks (sale, rflag, sku)
CREATE INDEX idx_jnl_sale_rflag_sku ON jnl(sale, rflag, sku);

-- Additional helpful indexes (if the composite ones above don't exist yet)
-- Index on jnh.tstamp for date filtering
CREATE INDEX idx_jnh_tstamp ON jnh(tstamp);

-- Index on jnh.sale for joins
CREATE INDEX idx_jnh_sale ON jnh(sale);

-- Index on jnl.sale for general joins
CREATE INDEX idx_jnl_sale ON jnl(sale);

CREATE INDEX idx_jnl_date_rflag_sku ON jnl(date, rflag, sku);

