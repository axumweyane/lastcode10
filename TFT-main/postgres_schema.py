"""
PostgreSQL Database Schema for TFT Stock Prediction System
Creates the required tables and indexes for the TFT system
"""

CREATE_SCHEMA_SQL = """
-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Set search path
SET search_path TO public;

-- Create symbols table for metadata
CREATE TABLE IF NOT EXISTS symbols (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(20),
    market_cap BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create OHLCV data table (main price data)
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    volume BIGINT,
    adj_open NUMERIC(12,4),
    adj_high NUMERIC(12,4),
    adj_low NUMERIC(12,4),
    adj_close NUMERIC(12,4),
    adj_volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- Create fundamentals table
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    market_cap BIGINT,
    pe_ratio NUMERIC(10,2),
    eps NUMERIC(10,4),
    dividend_yield NUMERIC(6,4),
    book_value NUMERIC(12,4),
    debt_to_equity NUMERIC(10,2),
    roe NUMERIC(6,4),
    roa NUMERIC(6,4),
    revenue BIGINT,
    net_income BIGINT,
    shares_outstanding BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- Create sentiment data table
CREATE TABLE IF NOT EXISTS sentiment (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sentiment_score NUMERIC(5,4), -- Score between -1 and 1
    sentiment_magnitude NUMERIC(5,4), -- Magnitude 0 to 1
    news_count INTEGER DEFAULT 0,
    reddit_mentions INTEGER DEFAULT 0,
    reddit_sentiment NUMERIC(5,4),
    twitter_mentions INTEGER DEFAULT 0,
    twitter_sentiment NUMERIC(5,4),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date, source)
);

-- Create earnings calendar table
CREATE TABLE IF NOT EXISTS earnings (
    symbol VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_estimate NUMERIC(10,4),
    earnings_actual NUMERIC(10,4),
    surprise_percent NUMERIC(6,2),
    announcement_time VARCHAR(20), -- 'BMO' (before market open) or 'AMC' (after market close)
    fiscal_quarter VARCHAR(10),
    fiscal_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, earnings_date)
);

-- Create economic indicators table (for macro features)
CREATE TABLE IF NOT EXISTS economic_indicators (
    indicator_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value NUMERIC(15,6),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_name, date)
);

-- Create VIX data table (for market regime detection)
CREATE TABLE IF NOT EXISTS vix_data (
    date DATE PRIMARY KEY,
    vix_open NUMERIC(6,2),
    vix_high NUMERIC(6,2),
    vix_low NUMERIC(6,2),
    vix_close NUMERIC(6,2),
    vix_volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model predictions table (for tracking predictions)
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    prediction_value NUMERIC(10,6),
    prediction_type VARCHAR(20), -- 'returns', 'classification', 'quintile'
    confidence_lower NUMERIC(10,6),
    confidence_upper NUMERIC(10,6),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model performance tracking table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC(10,6),
    test_start_date DATE,
    test_end_date DATE,
    symbols_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create training jobs table (for tracking model training)
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(100),
    symbols TEXT[], -- Array of symbols
    start_date DATE,
    end_date DATE,
    target_type VARCHAR(20),
    config JSONB, -- Store training configuration
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    final_loss NUMERIC(10,6),
    epochs_trained INTEGER,
    model_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper trading risk reports (HI-1 fix: risk manager wiring)
CREATE TABLE IF NOT EXISTS paper_risk_reports (
    id              SERIAL PRIMARY KEY,
    report_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drawdown        DOUBLE PRECISION,
    var_99          DOUBLE PRECISION,
    cvar_95         DOUBLE PRECISION,
    sharpe_21       DOUBLE PRECISION,
    killed_strategies TEXT[] DEFAULT '{}',
    correlation_alerts JSONB DEFAULT '[]',
    portfolio_breached BOOLEAN DEFAULT FALSE,
    raw_json        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_prr_time ON paper_risk_reports(report_time);

-- Bayesian weight state (adaptive ensemble weights)
CREATE TABLE IF NOT EXISTS bayesian_weight_state (
    strategy_name   VARCHAR(64) PRIMARY KEY,
    alpha           DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    n_updates       INTEGER NOT NULL DEFAULT 0,
    state_json      JSONB DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- VWAP execution statistics (slippage tracking)
CREATE TABLE IF NOT EXISTS paper_execution_stats (
    id              SERIAL PRIMARY KEY,
    execution_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10) NOT NULL,
    side            VARCHAR(4) NOT NULL,
    total_requested DOUBLE PRECISION NOT NULL,
    total_filled    DOUBLE PRECISION NOT NULL DEFAULT 0,
    expected_price  DOUBLE PRECISION,
    filled_avg_price DOUBLE PRECISION,
    slippage_bps    DOUBLE PRECISION DEFAULT 0,
    fill_rate       DOUBLE PRECISION DEFAULT 0,
    num_slices      INTEGER DEFAULT 0,
    used_fallback   BOOLEAN DEFAULT FALSE,
    adv_capped      BOOLEAN DEFAULT FALSE,
    elapsed_s       DOUBLE PRECISION DEFAULT 0,
    slices_json     JSONB DEFAULT '[]',
    error           TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_exec_stats_time ON paper_execution_stats(execution_time);
CREATE INDEX IF NOT EXISTS idx_exec_stats_ticker ON paper_execution_stats(ticker);

-- LLM signal analyses (daily analyst reports)
CREATE TABLE IF NOT EXISTS paper_signal_analyses (
    id              SERIAL PRIMARY KEY,
    analysis_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime          VARCHAR(32),
    summary         TEXT NOT NULL,
    patterns        TEXT,
    confidence      VARCHAR(8),
    flags_json      JSONB DEFAULT '{}',
    top_signals     JSONB DEFAULT '[]',
    model_used      VARCHAR(64),
    latency_s       DOUBLE PRECISION DEFAULT 0,
    raw_json        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_signal_analyses_time ON paper_signal_analyses(analysis_time);

-- Dead letter queue (failed Kafka message persistence + retry)
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id              SERIAL PRIMARY KEY,
    service_name    VARCHAR(128) NOT NULL,
    topic           VARCHAR(256) NOT NULL,
    message_key     TEXT,
    message_value   JSONB NOT NULL DEFAULT '{}',
    error           TEXT NOT NULL DEFAULT '',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 5,
    next_retry_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          VARCHAR(16) NOT NULL DEFAULT 'PENDING'
);

CREATE INDEX IF NOT EXISTS idx_dlq_status_next_retry
    ON dead_letter_queue(status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_dlq_service
    ON dead_letter_queue(service_name);
CREATE INDEX IF NOT EXISTS idx_dlq_created
    ON dead_letter_queue(created_at);

-- =============================================
-- TimescaleDB: Hypertable conversions
-- =============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Convert OHLCV to hypertable (partitioned by date)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'ohlcv'
    ) THEN
        PERFORM create_hypertable('ohlcv', 'date', migrate_data => true);
    END IF;
END $$;

-- Convert paper_risk_reports to hypertable (partitioned by report_time)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'paper_risk_reports'
    ) THEN
        PERFORM create_hypertable('paper_risk_reports', 'report_time', migrate_data => true);
    END IF;
END $$;

-- Convert paper_execution_stats to hypertable (partitioned by execution_time)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'paper_execution_stats'
    ) THEN
        PERFORM create_hypertable('paper_execution_stats', 'execution_time', migrate_data => true);
    END IF;
END $$;

-- Convert paper_signal_analyses to hypertable (partitioned by analysis_time)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'paper_signal_analyses'
    ) THEN
        PERFORM create_hypertable('paper_signal_analyses', 'analysis_time', migrate_data => true);
    END IF;
END $$;

-- =============================================
-- TimescaleDB: Retention policies
-- =============================================

-- OHLCV: retain 365 days
SELECT add_retention_policy('ohlcv', INTERVAL '365 days', if_not_exists => true);

-- Paper risk reports: retain 90 days
SELECT add_retention_policy('paper_risk_reports', INTERVAL '90 days', if_not_exists => true);

-- Paper execution stats: retain 90 days
SELECT add_retention_policy('paper_execution_stats', INTERVAL '90 days', if_not_exists => true);

-- Paper signal analyses: retain 90 days
SELECT add_retention_policy('paper_signal_analyses', INTERVAL '90 days', if_not_exists => true);

-- =============================================
-- TimescaleDB: Continuous aggregates
-- =============================================

-- 15-minute OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15m
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('15 minutes', date) AS bucket,
    first(open, date)  AS open,
    max(high)           AS high,
    min(low)            AS low,
    last(close, date)  AS close,
    sum(volume)         AS volume
FROM ohlcv
GROUP BY symbol, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_15m',
    start_offset  => INTERVAL '3 days',
    end_offset    => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => true
);

-- 1-hour OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', date) AS bucket,
    first(open, date)  AS open,
    max(high)           AS high,
    min(low)            AS low,
    last(close, date)  AS close,
    sum(volume)         AS volume
FROM ohlcv
GROUP BY symbol, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset  => INTERVAL '7 days',
    end_offset    => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => true
);

-- 1-day OHLCV aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1d
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', date) AS bucket,
    first(open, date)  AS open,
    max(high)           AS high,
    min(low)            AS low,
    last(close, date)  AS close,
    sum(volume)         AS volume
FROM ohlcv
GROUP BY symbol, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_1d',
    start_offset  => INTERVAL '30 days',
    end_offset    => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => true
);

-- Create indexes for performance optimization

-- OHLCV indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv(symbol, date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol);

-- Fundamentals indexes
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date ON fundamentals(symbol, date);
CREATE INDEX IF NOT EXISTS idx_fundamentals_date ON fundamentals(date);

-- Sentiment indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_date ON sentiment(symbol, date);
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment(date);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment(source);

-- Earnings indexes
CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date ON earnings(symbol, earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings(earnings_date);

-- Economic indicators indexes
CREATE INDEX IF NOT EXISTS idx_economic_indicators_name_date ON economic_indicators(indicator_name, date);
CREATE INDEX IF NOT EXISTS idx_economic_indicators_date ON economic_indicators(date);

-- VIX data index
CREATE INDEX IF NOT EXISTS idx_vix_data_date ON vix_data(date);

-- Model predictions indexes
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_pred_date ON model_predictions(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON model_predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON model_predictions(model_version);

-- Model performance indexes
CREATE INDEX IF NOT EXISTS idx_performance_model_version ON model_performance(model_version);
CREATE INDEX IF NOT EXISTS idx_performance_evaluation_date ON model_performance(evaluation_date);

-- Training jobs indexes
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at);

-- Create triggers for updating updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to symbols table
CREATE TRIGGER update_symbols_updated_at BEFORE UPDATE ON symbols
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to earnings table
CREATE TRIGGER update_earnings_updated_at BEFORE UPDATE ON earnings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- Latest fundamentals view
CREATE OR REPLACE VIEW latest_fundamentals AS
SELECT DISTINCT ON (symbol) 
    symbol, date, market_cap, pe_ratio, eps, dividend_yield,
    book_value, debt_to_equity, roe, roa, revenue, net_income
FROM fundamentals
ORDER BY symbol, date DESC;

-- Daily sentiment aggregated view
CREATE OR REPLACE VIEW daily_sentiment_agg AS
SELECT 
    symbol,
    date,
    AVG(sentiment_score) as avg_sentiment,
    SUM(news_count) as total_news,
    COUNT(*) as source_count
FROM sentiment
GROUP BY symbol, date;

-- Recent earnings view (next 90 days)
CREATE OR REPLACE VIEW upcoming_earnings AS
SELECT 
    symbol, earnings_date, earnings_estimate,
    EXTRACT(DOW FROM earnings_date) as day_of_week,
    earnings_date - CURRENT_DATE as days_until
FROM earnings
WHERE earnings_date >= CURRENT_DATE 
    AND earnings_date <= CURRENT_DATE + INTERVAL '90 days'
ORDER BY earnings_date;

-- Model performance summary view
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model_version,
    MAX(evaluation_date) as latest_evaluation,
    AVG(CASE WHEN metric_name = 'mse' THEN metric_value END) as avg_mse,
    AVG(CASE WHEN metric_name = 'mae' THEN metric_value END) as avg_mae,
    AVG(CASE WHEN metric_name = 'r2_score' THEN metric_value END) as avg_r2,
    AVG(CASE WHEN metric_name = 'directional_accuracy' THEN metric_value END) as avg_directional_accuracy
FROM model_performance
GROUP BY model_version;

-- Insert some initial economic indicators that we might track
INSERT INTO economic_indicators (indicator_name, date, value, source) VALUES
('VIX', CURRENT_DATE, 20.0, 'CBOE'),
('10Y_TREASURY', CURRENT_DATE, 4.5, 'FRED'),
('UNEMPLOYMENT_RATE', CURRENT_DATE, 3.7, 'BLS'),
('GDP_GROWTH', CURRENT_DATE, 2.1, 'BEA')
ON CONFLICT (indicator_name, date) DO NOTHING;

-- Create function to get data quality metrics
CREATE OR REPLACE FUNCTION get_data_quality_metrics(
    p_symbol VARCHAR(10),
    p_start_date DATE,
    p_end_date DATE
)
RETURNS TABLE(
    symbol VARCHAR(10),
    total_records BIGINT,
    missing_prices BIGINT,
    zero_volume_days BIGINT,
    data_completeness NUMERIC(5,4),
    date_range_start DATE,
    date_range_end DATE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_symbol as symbol,
        COUNT(*) as total_records,
        COUNT(CASE WHEN adj_close IS NULL THEN 1 END) as missing_prices,
        COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_days,
        CASE 
            WHEN COUNT(*) = 0 THEN 0 
            ELSE 1 - (COUNT(CASE WHEN adj_close IS NULL THEN 1 END)::numeric / COUNT(*)::numeric)
        END as data_completeness,
        MIN(date) as date_range_start,
        MAX(date) as date_range_end
    FROM ohlcv
    WHERE ohlcv.symbol = p_symbol
        AND date BETWEEN p_start_date AND p_end_date;
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate technical indicators (basic RSI example)
CREATE OR REPLACE FUNCTION calculate_rsi(
    p_symbol VARCHAR(10),
    p_period INTEGER DEFAULT 14
)
RETURNS TABLE(
    symbol VARCHAR(10),
    date DATE,
    rsi NUMERIC(6,2)
) AS $$
BEGIN
    RETURN QUERY
    WITH price_changes AS (
        SELECT 
            ohlcv.symbol,
            ohlcv.date,
            adj_close,
            adj_close - LAG(adj_close) OVER (PARTITION BY ohlcv.symbol ORDER BY date) as price_change
        FROM ohlcv
        WHERE ohlcv.symbol = p_symbol
        ORDER BY date
    ),
    gains_losses AS (
        SELECT 
            symbol,
            date,
            CASE WHEN price_change > 0 THEN price_change ELSE 0 END as gain,
            CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END as loss
        FROM price_changes
        WHERE price_change IS NOT NULL
    ),
    average_gains_losses AS (
        SELECT 
            symbol,
            date,
            AVG(gain) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) as avg_gain,
            AVG(loss) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) as avg_loss
        FROM gains_losses
    )
    SELECT 
        agl.symbol,
        agl.date,
        CASE 
            WHEN avg_loss = 0 THEN 100
            ELSE 100 - (100 / (1 + (avg_gain / avg_loss)))
        END as rsi
    FROM average_gains_losses agl;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

COMMIT;
"""

SAMPLE_DATA_SQL = """
-- Sample data insertion script
-- This would typically be populated by your data pipeline

-- Sample symbols
INSERT INTO symbols (symbol, company_name, sector, industry, exchange) VALUES
('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 'NASDAQ'),
('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Content & Information', 'NASDAQ'),
('MSFT', 'Microsoft Corporation', 'Technology', 'Software', 'NASDAQ'),
('TSLA', 'Tesla, Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'NASDAQ'),
('AMZN', 'Amazon.com, Inc.', 'Consumer Cyclical', 'Internet Retail', 'NASDAQ')
ON CONFLICT (symbol) DO UPDATE SET
    company_name = EXCLUDED.company_name,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    exchange = EXCLUDED.exchange,
    updated_at = CURRENT_TIMESTAMP;

-- Sample OHLCV data (you would populate this with real data)
-- This is just a template - replace with actual data loading
/*
INSERT INTO ohlcv (symbol, date, open, high, low, close, volume, adj_open, adj_high, adj_low, adj_close, adj_volume)
SELECT 
    symbol,
    generate_series(
        CURRENT_DATE - INTERVAL '2 years',
        CURRENT_DATE,
        INTERVAL '1 day'
    )::date as date,
    100 + random() * 50 as open,
    105 + random() * 50 as high,
    95 + random() * 50 as low,
    100 + random() * 50 as close,
    (1000000 + random() * 9000000)::bigint as volume,
    100 + random() * 50 as adj_open,
    105 + random() * 50 as adj_high,
    95 + random() * 50 as adj_low,
    100 + random() * 50 as adj_close,
    (1000000 + random() * 9000000)::bigint as adj_volume
FROM symbols
WHERE EXTRACT(DOW FROM generate_series(
    CURRENT_DATE - INTERVAL '2 years',
    CURRENT_DATE,
    INTERVAL '1 day'
)) NOT IN (0, 6); -- Exclude weekends
*/

COMMIT;
"""

if __name__ == "__main__":
    import psycopg2
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Database configuration
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'stock_trading_analysis'),
        'user': os.getenv('POSTGRES_USER', 'trading_user'),
        'password': os.environ['POSTGRES_PASSWORD'],
        'port': int(os.getenv('POSTGRES_PORT', 5432))
    }
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        
        print("Connected to PostgreSQL database")
        
        # Execute schema creation
        with conn.cursor() as cursor:
            print("Creating database schema...")
            cursor.execute(CREATE_SCHEMA_SQL)
            print("Database schema created successfully")
            
            # Optionally create sample data
            create_sample = input("Create sample symbols data? (y/n): ").lower().strip()
            if create_sample == 'y':
                cursor.execute(SAMPLE_DATA_SQL)
                print("Sample data created")
        
        conn.close()
        print("Database setup completed successfully")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        import traceback
        traceback.print_exc()
