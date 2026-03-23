#!/usr/bin/env python3
"""
Populate missing TimescaleDB tables: vix_data, fundamentals, sentiment.

Usage:
    python scripts/populate_tables.py
    python scripts/populate_tables.py --tables vix fundamentals sentiment
"""

import argparse
import os
import sys
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

SEP = "=" * 70
THIN = "-" * 70

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "XOM"]
START_DATE = "2024-01-01"


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "15432")),
        dbname=os.getenv("DB_NAME", "apex"),
        user=os.getenv("DB_USER", "apex_user"),
        password=os.getenv("DB_PASSWORD", "apex_pass"),
    )


def create_tables(conn):
    """Create tables if they don't exist."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vix_data (
            date DATE PRIMARY KEY,
            vix_open NUMERIC(6,2),
            vix_high NUMERIC(6,2),
            vix_low NUMERIC(6,2),
            vix_close NUMERIC(6,2),
            vix_volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
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
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            sentiment_score NUMERIC(5,4),
            sentiment_magnitude NUMERIC(5,4),
            news_count INTEGER DEFAULT 0,
            reddit_mentions INTEGER DEFAULT 0,
            reddit_sentiment NUMERIC(5,4),
            twitter_mentions INTEGER DEFAULT 0,
            twitter_sentiment NUMERIC(5,4),
            source VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date, source)
        )
    """)

    conn.commit()
    cur.close()
    print("  Tables created/verified")


# ---------------------------------------------------------------------------
# VIX Data
# ---------------------------------------------------------------------------

def populate_vix(conn):
    """Download VIX history from yfinance and insert into vix_data."""
    print(f"\n{THIN}")
    print("  POPULATING: vix_data")
    print(THIN)

    ticker = yf.Ticker("^VIX")
    hist = ticker.history(start=START_DATE, auto_adjust=True)

    if hist.empty:
        print("  FAIL: No VIX data downloaded")
        return 0

    print(f"  Downloaded {len(hist)} VIX rows ({hist.index[0].date()} to {hist.index[-1].date()})")

    rows = []
    for dt, row in hist.iterrows():
        d = dt.tz_localize(None).date() if dt.tzinfo else dt.date()
        rows.append((
            d,
            round(float(row["Open"]), 2),
            round(float(row["High"]), 2),
            round(float(row["Low"]), 2),
            round(float(row["Close"]), 2),
            int(row.get("Volume", 0)),
        ))

    cur = conn.cursor()
    execute_values(
        cur,
        """
        INSERT INTO vix_data (date, vix_open, vix_high, vix_low, vix_close, vix_volume)
        VALUES %s
        ON CONFLICT (date) DO UPDATE SET
            vix_open = EXCLUDED.vix_open,
            vix_high = EXCLUDED.vix_high,
            vix_low = EXCLUDED.vix_low,
            vix_close = EXCLUDED.vix_close,
            vix_volume = EXCLUDED.vix_volume
        """,
        rows,
    )
    conn.commit()
    cur.close()

    print(f"  Inserted {len(rows)} VIX rows")
    return len(rows)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def populate_fundamentals(conn):
    """Pull fundamentals from yfinance ticker.info for each symbol."""
    print(f"\n{THIN}")
    print("  POPULATING: fundamentals")
    print(THIN)

    today = date.today()
    # Generate one row per symbol per trading day in the OHLCV date range
    # Fundamentals change slowly, so we backfill the same values across all dates
    cur = conn.cursor()

    total_inserted = 0
    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            eps = info.get("trailingEps")
            dividend_yield = info.get("dividendYield")
            book_value = info.get("bookValue")
            debt_to_equity = info.get("debtToEquity")
            roe = info.get("returnOnEquity")
            roa = info.get("returnOnAssets")
            revenue = info.get("totalRevenue")
            net_income = info.get("netIncomeToCommon")
            shares = info.get("sharesOutstanding")

            if pe_ratio is not None:
                pe_ratio = round(float(pe_ratio), 2)
            if eps is not None:
                eps = round(float(eps), 4)
            if dividend_yield is not None:
                dividend_yield = round(float(dividend_yield), 4)
            if book_value is not None:
                book_value = round(float(book_value), 4)
            if debt_to_equity is not None:
                debt_to_equity = round(float(debt_to_equity), 2)
            if roe is not None:
                roe = round(float(roe), 4)
            if roa is not None:
                roa = round(float(roa), 4)

            # Get all dates this symbol has OHLCV data
            cur.execute(
                "SELECT DISTINCT date FROM ohlcv WHERE symbol = %s ORDER BY date",
                (symbol,),
            )
            ohlcv_dates = [r[0] for r in cur.fetchall()]

            if not ohlcv_dates:
                # Use business day range as fallback
                ohlcv_dates = [d.date() for d in pd.bdate_range(START_DATE, today)]

            rows = []
            for d in ohlcv_dates:
                rows.append((
                    symbol, d, market_cap, pe_ratio, eps, dividend_yield,
                    book_value, debt_to_equity, roe, roa, revenue, net_income, shares,
                ))

            execute_values(
                cur,
                """
                INSERT INTO fundamentals (
                    symbol, date, market_cap, pe_ratio, eps, dividend_yield,
                    book_value, debt_to_equity, roe, roa, revenue, net_income, shares_outstanding
                ) VALUES %s
                ON CONFLICT (symbol, date) DO UPDATE SET
                    market_cap = EXCLUDED.market_cap,
                    pe_ratio = EXCLUDED.pe_ratio,
                    eps = EXCLUDED.eps,
                    dividend_yield = EXCLUDED.dividend_yield,
                    book_value = EXCLUDED.book_value,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    roe = EXCLUDED.roe,
                    roa = EXCLUDED.roa,
                    revenue = EXCLUDED.revenue,
                    net_income = EXCLUDED.net_income,
                    shares_outstanding = EXCLUDED.shares_outstanding
                """,
                rows,
            )
            conn.commit()
            total_inserted += len(rows)
            print(f"OK ({len(rows)} rows, PE={pe_ratio}, EPS={eps})")

        except Exception as e:
            print(f"FAIL: {e}")
            conn.rollback()

    cur.close()
    print(f"  Total: {total_inserted} fundamental rows inserted")
    return total_inserted


# ---------------------------------------------------------------------------
# Sentiment (VADER)
# ---------------------------------------------------------------------------

def populate_sentiment(conn):
    """Pull news headlines from yfinance, score with VADER, insert into sentiment."""
    print(f"\n{THIN}")
    print("  POPULATING: sentiment")
    print(THIN)

    # Install/import VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("  Installing vaderSentiment...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment", "-q"])
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    cur = conn.cursor()
    total_inserted = 0

    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                print("no news")
                continue

            # Group headlines by date
            date_headlines = {}
            for item in news:
                # yfinance news items have 'providerPublishTime' (unix timestamp)
                # or 'content' dict with 'pubDate'
                ts = None
                title = None

                if isinstance(item, dict):
                    # Try different formats
                    ts = item.get("providerPublishTime")
                    title = item.get("title")

                    if ts is None and "content" in item:
                        content = item["content"]
                        title = content.get("title", title)
                        pub_date = content.get("pubDate")
                        if pub_date:
                            try:
                                ts = int(datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp())
                            except Exception:
                                pass

                if ts is None or title is None:
                    continue

                d = datetime.fromtimestamp(ts).date()
                if d not in date_headlines:
                    date_headlines[d] = []
                date_headlines[d].append(title)

            if not date_headlines:
                print("no parseable news")
                continue

            rows = []
            for d, headlines in sorted(date_headlines.items()):
                scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
                avg_score = np.mean(scores)
                magnitude = np.mean([abs(s) for s in scores])

                rows.append((
                    symbol,
                    d,
                    round(float(avg_score), 4),
                    round(float(magnitude), 4),
                    len(headlines),
                    0,      # reddit_mentions
                    None,   # reddit_sentiment
                    0,      # twitter_mentions
                    None,   # twitter_sentiment
                    "yfinance_vader",
                ))

            if rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO sentiment (
                        symbol, date, sentiment_score, sentiment_magnitude,
                        news_count, reddit_mentions, reddit_sentiment,
                        twitter_mentions, twitter_sentiment, source
                    ) VALUES %s
                    ON CONFLICT (symbol, date, source) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_magnitude = EXCLUDED.sentiment_magnitude,
                        news_count = EXCLUDED.news_count
                    """,
                    rows,
                )
                conn.commit()
                total_inserted += len(rows)
                print(f"OK ({len(rows)} days, {sum(len(h) for h in date_headlines.values())} headlines, avg={np.mean([r[2] for r in rows]):.3f})")
            else:
                print("no rows generated")

        except Exception as e:
            print(f"FAIL: {e}")
            conn.rollback()

    cur.close()
    print(f"  Total: {total_inserted} sentiment rows inserted")
    return total_inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Populate missing TimescaleDB tables")
    parser.add_argument("--tables", nargs="+", default=["vix", "fundamentals", "sentiment"],
                        choices=["vix", "fundamentals", "sentiment"])
    args = parser.parse_args()

    print(SEP)
    print("  APEX DATA POPULATION")
    print(f"  DB: {os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '15432')}/{os.getenv('DB_NAME', 'apex')}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Start date: {START_DATE}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)

    conn = get_connection()
    create_tables(conn)

    results = {}

    if "vix" in args.tables:
        results["vix_data"] = populate_vix(conn)

    if "fundamentals" in args.tables:
        results["fundamentals"] = populate_fundamentals(conn)

    if "sentiment" in args.tables:
        results["sentiment"] = populate_sentiment(conn)

    conn.close()

    print(f"\n{SEP}")
    print("  POPULATION COMPLETE")
    for table, count in results.items():
        print(f"  {table}: {count} rows")
    print(SEP)


if __name__ == "__main__":
    main()
