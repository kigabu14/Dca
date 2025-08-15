import os
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text
from passlib.hash import bcrypt

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# ---------------- App Config ----------------
st.set_page_config(page_title="Plug2Plug DCA Pro", page_icon="🧠", layout="wide")
DB_URL = "sqlite:///portfolio_users.db"

# ---------------- Secrets / API Keys ----------------
# อ่านจาก st.secrets ก่อน แล้ว fallback ไป ENV เพื่อกันคีย์หาย
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------- DB Setup ----------------
engine = create_engine(DB_URL, future=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,     -- 'US' or 'TH'
            qty REAL NOT NULL,
            price REAL NOT NULL,
            trade_date TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dividends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            ex_date TEXT NOT NULL,         -- Ex-div date (ISO)
            amount REAL NOT NULL,          -- per share
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, symbol, market, ex_date)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            line_token TEXT,
            telegram_token TEXT,
            telegram_chat_id TEXT,
            notify_on_buy INTEGER DEFAULT 0   -- 0/1
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            sent_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """))

init_db()

# ---------------- Helpers ----------------
def valid_username(u: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", u or ""))

def yf_symbol(symbol: str, market: str):
    return symbol if market == "US" else f"{symbol}.BK"

@st.cache_data(ttl=300)
def get_price(symbol: str, market: str):
    t = yf.Ticker(yf_symbol(symbol, market))
    price = None
    try:
        info = getattr(t, "fast_info", {}) or {}
        price = float(info.get("last_price")) if info else None
    except Exception:
        price = None
    if price is None:
        hist = t.history(period="5d", interval="1d")
        price = float(hist["Close"].iloc[-1]) if len(hist) else np.nan
    return price

@st.cache_data(ttl=300)
def get_hist(symbol: str, market: str, period="6mo", with_hlc=True):
    t = yf.Ticker(yf_symbol(symbol, market))
    df = t.history(period=period, interval="1d")
    if df is None or df.empty:
        return df
    if with_hlc:
        for c in ["High","Low"]:
            if c not in df.columns:
                df[c] = np.nan
    if "Close" not in df.columns:
        df["Close"] = np.nan
    return df

def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=period-1, adjust=False).mean()
    roll_down = down.ewm(com=period-1, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def load_trades(user_id: int, symbol: str = None):
    q = "SELECT * FROM trades WHERE user_id=:uid"
    params = {"uid": user_id}
    if symbol:
        q += " AND symbol=:s"
        params["s"] = symbol
    q += " ORDER BY trade_date ASC"
    with engine.begin() as conn:
        df = pd.read_sql(text(q), conn, params=params)
    return df

def add_trade(user_id, symbol, market, qty, price, trade_date):
    with engine.begin() as conn:
        conn.execute(
            text("""
                 INSERT INTO trades(user_id, symbol, market, qty, price, trade_date)
                 VALUES(:uid,:s,:m,:q,:p,:d)
            """),
            {"uid": user_id, "s": symbol, "m": market, "q": qty, "p": price, "d": trade_date}
        )

def delete_trade(user_id, trade_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM trades WHERE id=:id AND user_id=:uid"),
                     {"id": trade_id, "uid": user_id})

def portfolio_symbols(user_id: int):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT symbol, market FROM trades WHERE user_id=:uid
        """), {"uid": user_id}).fetchall()
    return [{"symbol": r.symbol, "market": r.market} for r in rows]

def fetch_and_store_dividends(user_id: int, symbol: str, market: str, years=5):
    t = yf.Ticker(yf_symbol(symbol, market))
    try:
        div = t.dividends  # Series index = ex-date (อาจมี tz)
    except Exception:
        div = None
    if div is None or len(div) == 0:
        return 0

    # ทำ index เป็น tz-naive ก่อน filter
    idx = pd.to_datetime(div.index)
    try:
        idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    div.index = idx

    cutoff = pd.Timestamp.today().tz_localize(None) - pd.DateOffset(years=years)
    div = div[div.index >= cutoff]

    inserted = 0
    with engine.begin() as conn:
        for dt_idx, amt in div.items():
            ex_date = pd.to_datetime(dt_idx).date().isoformat()
            try:
                conn.execute(text("""
                    INSERT OR IGNORE INTO dividends(user_id, symbol, market, ex_date, amount)
                    VALUES(:uid,:s,:m,:d,:a)
                """), {"uid": user_id, "s": symbol, "m": market, "d": ex_date, "a": float(amt)})
                inserted += 1
            except Exception:
                pass
    return inserted

def load_dividends(user_id: int, symbol: str = None):
    q = "SELECT symbol, market, ex_date, amount FROM dividends WHERE user_id=:uid"
    params = {"uid": user_id}
    if symbol:
        q += " AND symbol=:s"
        params["s"] = symbol
    q += " ORDER BY ex_date DESC"
    with engine.begin() as conn:
        df = pd.read_sql(text(q), conn, params=params)
    return df

def ttm_dividend_per_share(user_id: int, symbol: str):
    # รวมปันผล 12 เดือนล่าสุด (TTM)
    one_year_ago = (datetime.utcnow() - timedelta(days=365)).date().isoformat()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT amount FROM dividends
            WHERE user_id=:uid AND symbol=:s AND ex_date>=:d
        """), {"uid": user_id, "s": symbol, "d": one_year_ago}).fetchall()
    if not rows:
        return 0.0
    return float(np.sum([r[0] for r in rows]))

def portfolio_summary(user_id: int):
    # ดึงข้อมูลดิบ
    with engine.begin() as conn:
        raw = pd.read_sql(
            text("SELECT symbol, market, qty, price FROM trades WHERE user_id=:uid"),
            conn, params={"uid": user_id}
        )

    # ถ้าไม่มีดีลก็คืน DataFrame โครงสร้างพร้อมแสดง
    if raw.empty:
        return pd.DataFrame(columns=[
            "symbol","market","units","avg_cost","last","pnl_%","pnl_value",
            "ttm_div_ps","yoc_%","ttm_div_total"
        ])

    # ทำความสะอาด dtype
    for c in ["qty","price"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

    # คำนวณ units / avg_cost แบบ aggregate (เลี่ยง FutureWarning)
    raw["cost"] = raw["qty"] * raw["price"]
    grouped = (raw.groupby(["symbol","market"], as_index=False)
                  .agg(units=("qty","sum"), total_cost=("cost","sum")))
    grouped["avg_cost"] = np.where(grouped["units"] > 0,
                                   grouped["total_cost"] / grouped["units"],
                                   0.0)
    out = grouped.drop(columns=["total_cost"])

    # ราคาล่าสุด + ปันผล TTM
    prices, ttm_ps = [], []
    for _, row in out.iterrows():
        last = get_price(row["symbol"], row["market"])
        prices.append(last if pd.notna(last) else 0.0)
        ttm_ps.append(ttm_dividend_per_share(user_id, row["symbol"]))

    out["last"] = prices
    out["pnl_%"] = np.where(out["avg_cost"] > 0,
                            (out["last"] - out["avg_cost"]) / out["avg_cost"] * 100,
                            0.0)
    out["pnl_value"] = (out["last"] - out["avg_cost"]) * out["units"]
    out["ttm_div_ps"] = ttm_ps
    out["yoc_%"] = np.where(out["avg_cost"] > 0, out["ttm_div_ps"] / out["avg_cost"] * 100, 0.0)
    out["ttm_div_total"] = out["ttm_div_ps"] * out["units"]

    num_cols = ["units","avg_cost","last","pnl_%","pnl_value","ttm_div_ps","yoc_%","ttm_div_total"]
    out[num_cols] = out[num_cols].fillna(0.0)

    return out.sort_values("symbol")

# ----------- Notifications -----------
def get_user_settings(user_id: int):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT user_id, line_token, telegram_token, telegram_chat_id, notify_on_buy
            FROM user_settings WHERE user_id=:uid
        """), {"uid": user_id}).fetchone()
    if not row:
        return {"user_id": user_id, "line_token": None, "telegram_token": None, "telegram_chat_id": None, "notify_on_buy": 0}
    return dict(row._mapping)

def save_user_settings(user_id: int, line_token: str, telegram_token: str, telegram_chat_id: str, notify_on_buy: bool):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_settings(user_id, line_token, telegram_token, telegram_chat_id, notify_on_buy)
            VALUES(:uid,:l,:t,:c,:n)
            ON CONFLICT(user_id) DO UPDATE SET
                line_token=excluded.line_token,
                telegram_token=excluded.telegram_token,
                telegram_chat_id=excluded.telegram_chat_id,
                notify_on_buy=excluded.notify_on_buy
        """), {"uid": user_id, "l": line_token, "t": telegram_token, "c": telegram_chat_id, "n": 1 if notify_on_buy else 0})

def already_alerted_recently(user_id: int, symbol: str, action: str, hours=12):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id FROM alerts
            WHERE user_id=:uid AND symbol=:s AND action=:a AND sent_at>=:cut
            ORDER BY sent_at DESC LIMIT 1
        """), {"uid": user_id, "s": symbol, "a": action, "cut": cutoff}).fetchone()
    return row is not None

def record_alert(user_id: int, symbol: str, action: str):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO alerts(user_id, symbol, action) VALUES(:uid,:s,:a)"),
                     {"uid": user_id, "s": symbol, "a": action})

def line_notify(token: str, message: str):
    try:
        r = requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message}, timeout=8
        )
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)

def telegram_notify(bot_token: str, chat_id: str, message: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=8)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)

# ----------- Advice & Position Sizing -----------
def rule_based_advice(avg_cost: float, last: float, hist: pd.DataFrame, budget_month: float, lots: int):
    notes = []
    action = "HOLD"
    buy_qty = 0

    if np.isnan(last) or avg_cost <= 0 or hist is None or hist.empty:
        return action, buy_qty, ["ข้อมูลราคา/ต้นทุน/กราฟ ไม่พร้อม"], {"RSI14": np.nan, "ATR14": np.nan, "SMA20": np.nan, "SMA50": np.nan}

    hist = hist.copy()
    hist["SMA20"] = sma(hist["Close"], 20)
    hist["SMA50"] = sma(hist["Close"], 50)
    hist["EMA20"] = ema(hist["Close"], 20)
    hist["EMA50"] = ema(hist["Close"], 50)
    hist["RSI14"] = rsi(hist["Close"], 14)
    hist["ATR14"] = atr(hist, 14)

    c = hist["Close"].iloc[-1]
    s20 = hist["SMA20"].iloc[-1]
    s50 = hist["SMA50"].iloc[-1]
    e20 = hist["EMA20"].iloc[-1]
    e50 = hist["EMA50"].iloc[-1]
    r14 = hist["RSI14"].iloc[-1]
    a14 = hist["ATR14"].iloc[-1]

    gap = (avg_cost - last) / max(avg_cost, 1e-9) * 100
    up_trend = (c > s20) and (s20 > s50) and (e20 > e50)

    if gap >= 5:
        action = "BUY (DCA)"
        notes.append(f"ราคาต่ำกว่าทุนเฉลี่ย ~{gap:.1f}%")
    elif up_trend and 40 <= r14 <= 70:
        action = "BUY SMALL"
        notes.append("แนวโน้มขาขึ้น (Close>SMA20>SMA50 & EMA20>EMA50) และ RSI กลางๆ")
    elif r14 < 30:
        action = "WATCHLIST"
        notes.append("RSI Oversold — รอสัญญาณเด้งยืนยันก่อน")
    else:
        action = "HOLD"
        notes.append("ยังไม่ชัด รอย่อหรือรอเบรกไฮ")

    per_lot_budget = max(budget_month / max(lots,1), 0)
    if action.startswith("BUY") and per_lot_budget > 0 and (not np.isnan(a14)) and a14 > 0:
        risk_per_share = a14 * 2  # 2*ATR เป็น stop เบื้องต้น
        est_shares = int(per_lot_budget // max(risk_per_share, 1e-9))
        if est_shares <= 0 and last > 0:
            est_shares = int(per_lot_budget // last)
            if est_shares == 0 and per_lot_budget >= last*0.6:
                est_shares = 1
        buy_qty = est_shares
        notes.append(f"ATR14≈{a14:.2f} → ความเสี่ยง/หุ้น≈{risk_per_share:.2f} | งบ/ไม้≈{per_lot_budget:,.0f} → แนะนำซื้อ {buy_qty} หุ้น")

    notes.append("ตั้งจุดตัดขาดทุนเชิงวินัย (เช่น หลุด SMA50 หรือ 2×ATR ต่ำกว่าจุดเข้า)")
    return action, buy_qty, notes, {"RSI14": float(r14), "ATR14": float(a14), "SMA20": float(s20), "SMA50": float(s50)}

def try_notify_buy(user_id: int, symbol: str, market: str, action: str, price: float):
    settings = get_user_settings(user_id)
    if not settings or not settings.get("notify_on_buy", 0):
        return
    if not action.startswith("BUY"):
        return
    if already_alerted_recently(user_id, symbol, action, hours=12):
        return
    msg = f"[Plug2Plug DCA] {symbol} ({market}) สัญญาณ {action} @ ~{price:.2f}"
    if settings.get("line_token"):
        line_notify(settings.get("line_token"), msg)
    if settings.get("telegram_token") and settings.get("telegram_chat_id"):
        telegram_notify(settings.get("telegram_token"), settings.get("telegram_chat_id"), msg)
    record_alert(user_id, symbol, action)

# ----------- Gemini LLM -----------
def summarize_portfolio_with_gemini(portfolio_df: pd.DataFrame, model_name="gemini-1.5-flash"):
    if not GEMINI_API_KEY:
        return "⚠️ ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน st.secrets หรือ ENV"
    try:
        data = portfolio_df.to_dict(orient="records")
        prompt = ("
        คุณคือผู้ช่วยนักวิเคราะห์การลงทุนแนว DCA ที่ทำงานตามข้อมูลจริง ไม่มโน ไม่เดา
        ให้คำแนะนำที่สอดคล้องกับวินัยการลงทุนและความเสี่ยง โดยใช้เฉพาะข้อมูลที่ให้เท่านั้น

        [สถานะพอร์ตของผู้ใช้]:
        {pf_json}

        [ตัวเลขเทคนิค (ล่าสุด)]:
        {tech_json}

        [ข่าว/โซเชียลล่าสุด] (ทั้งภาษาไทยและอังกฤษ):
        {news_bullets}

        [บันทึกเพิ่มเติมจากผู้ใช้]:
        {user_notes}

        กรุณาวิเคราะห์และสรุปเป็นหมวดหมู่:
        1) 📈 ควรถัวเพิ่ม: หุ้นใด และเพราะอะไร 
        - ให้เหตุผล เช่น ราคาอยู่ในโซนสะสม, เทรนด์ดี, ข่าวบวก, พื้นฐานแข็งแรง
        2) ✋ ถือรอดู: หุ้นใด และเพราะอะไร 
        เช่น ข่าวไม่ชัดเจน, รอดูสัญญาณเทคนิค, รอดูผลประกอบการ
        3) 📉 ควรขายตัดขาดทุน: หุ้นใด และเพราะอะไร 
        เช่น แนวโน้มธุรกิจถดถอย, ข่าวลบ, หลุดแนวรับสำคัญ
        4) 🌱 ตัวเลือกเติบโตใหม่ (ถ้ามีจากข่าว):
        ชื่อหุ้น + เหตุผลแบบสั้นว่าทำไมถึงน่าสนใจ
    
        ข้อกำหนด:
        - ถ้าข้อมูลไม่เพียงพอให้ระบุว่า 'ไม่มีข้อมูลเพียงพอ'
        - ห้ามรับรองผลตอบแทน
        - ให้ bullet กระชับ ชัดเจน เป็นภาษาไทย
        ")
        model = genai.GenerativeModel(model_name)
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"LLM Error: {e}"

# --------------- Ticker List (Dropdown) ----------------
def load_tickers_from(path: str, fallback: list):
    try:
        df = pd.read_csv(path)
        items = sorted(df["symbol"].dropna().astype(str).unique().tolist())
        return items if items else fallback
    except Exception:
        return fallback

DEFAULT_TH = ["KTB","KBANK","SCB","PTT","PTTEP","SCC","ADVANC","AOT","CPALL","TRUE","GULF","BDMS","IVL","KCE","TOP"]
DEFAULT_US = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO","COST"]

tickers_th = load_tickers_from("tickers_th.csv", DEFAULT_TH)
tickers_us = load_tickers_from("tickers_us.csv", DEFAULT_US)

# --------------- Auth Layer ----------------
def register_user(username: str, password: str):
    if not valid_username(username):
        return False, "ชื่อผู้ใช้ต้องเป็น a-z A-Z 0-9 . _ - ยาว 3-32 ตัวอักษร"
    if len(password) < 6:
        return False, "รหัสผ่านต้องอย่างน้อย 6 ตัวอักษร"
    pw_hash = bcrypt.hash(password)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO users(username, password_hash) VALUES(:u,:p)"),
                         {"u": username, "p": pw_hash})
        return True, "สมัครสมาชิกสำเร็จ"
    except Exception as e:
        msg = str(e)
        if "UNIQUE constraint failed: users.username" in msg:
            return False, "ชื่อนี้ถูกใช้แล้ว"
        return False, "สมัครไม่สำเร็จ"

def login_user(username: str, password: str):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, username, password_hash FROM users WHERE username=:u"),
                           {"u": username}).fetchone()
    if not row:
        return False, "ไม่พบบัญชีผู้ใช้"
    if not bcrypt.verify(password, row.password_hash):
        return False, "รหัสผ่านไม่ถูกต้อง"
    return True, {"user_id": row.id, "username": row.username}

def logout():
    for k in ["logged_in","user_id","username"]:
        if k in st.session_state:
            del st.session_state[k]
    st.cache_data.clear()

# --------------- UI ----------------
def login_screen():
    st.title("🔐 เข้าสู่ระบบ - Plug2Plug DCA Pro")
    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

    with tab_login:
        u = st.text_input("ชื่อผู้ใช้", key="login_u")
        p = st.text_input("รหัสผ่าน", type="password", key="login_p")
        if st.button("เข้าสู่ระบบ"):
            ok, res = login_user(u.strip(), p)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user_id = res["user_id"]
                st.session_state.username = res["username"]
                st.success("ล็อกอินสำเร็จ")
                st.rerun()
            else:
                st.error(res)

    with tab_register:
        u2 = st.text_input("ตั้งชื่อผู้ใช้ (3-32 ตัว a-z A-Z 0-9 . _ -)", key="reg_u")
        p2 = st.text_input("ตั้งรหัสผ่าน (≥6 ตัว)", type="password", key="reg_p")
        if st.button("สมัครสมาชิก"):
            ok, msg = register_user(u2.strip(), p2)
            if ok:
                st.success(msg + " กรุณาไปที่แท็บ 'เข้าสู่ระบบ'")
            else:
                st.error(msg)

def app_screen():
    st.sidebar.success(f"ล็อกอินแล้ว: {st.session_state.username}")
    if st.sidebar.button("ออกจากระบบ"):
        logout()
        st.rerun()

    st.title("📈 Plug2Plug DCA Pro — Per-User Portfolio + Dividends + Alerts + LLM")
    st.caption("ถัวอย่างมีวินัย • ปันผล • สัญญาณแจ้งเตือน • สรุปด้วย Gemini")

    # Sidebar settings
    st.sidebar.header("📊 DCA Settings")
    market = st.sidebar.selectbox("ตลาด", ["TH","US"], key="market")
    options = tickers_th if market == "TH" else tickers_us
    options_with_custom = ["-- เลือกจากลิสต์ --"] + options + ["(Custom)"]
    choice = st.sidebar.selectbox("เลือกหุ้น", options_with_custom, index=0, key="symbol_choice")

    if choice == "(Custom)":
        symbol = st.sidebar.text_input("พิมพ์สัญลักษณ์เอง", value=("KTB" if market=="TH" else "AAPL"), key="symbol_custom").upper()
    elif choice == "-- เลือกจากลิสต์ --":
        symbol = (options[0] if options else ("KTB" if market=="TH" else "AAPL"))
    else:
        symbol = choice

    budget = st.sidebar.number_input("งบ DCA ต่อเดือน", min_value=0.0, value=10000.0, step=1000.0, key="budget")
    lots = st.sidebar.number_input("จำนวนไม้/เดือน", min_value=1, value=4, step=1, key="lots")

    st.sidebar.header("🔔 การแจ้งเตือน")
    settings = get_user_settings(st.session_state.user_id)
    line_token = st.sidebar.text_input("LINE Notify Token", value=settings.get("line_token") or "", type="password")
    tg_token = st.sidebar.text_input("Telegram Bot Token", value=settings.get("telegram_token") or "", type="password")
    tg_chat = st.sidebar.text_input("Telegram Chat ID", value=settings.get("telegram_chat_id") or "")
    notify_flag = st.sidebar.checkbox("แจ้งเตือนเมื่อมีสัญญาณซื้อ (ป้องกันสแปมทุก 12 ชม.)", value=bool(settings.get("notify_on_buy", 0)))
    if st.sidebar.button("บันทึกการตั้งค่าแจ้งเตือน"):
        save_user_settings(st.session_state.user_id, line_token.strip() or None, tg_token.strip() or None, tg_chat.strip() or None, notify_flag)
        st.success("บันทึกการตั้งค่าแจ้งเตือนแล้ว")

    st.sidebar.header("🧠 Gemini")
    gemini_model = st.sidebar.text_input("Gemini Model", value="gemini-1.5-flash")
    if not GEMINI_API_KEY:
        st.sidebar.warning("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน .streamlit/secrets.toml หรือ ENV")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["พอร์ต","บันทึกซื้อถัว","ชาร์ตราคา","ปันผล/Dividends"])

    # -------- Tab 2: บันทึกซื้อถัว --------
    with tab2:
        st.subheader("บันทึกการซื้อ (DCA)")
        c1, c2, c3 = st.columns(3)
        qty = c1.number_input("จำนวนหุ้น", min_value=0.0, value=100.0, step=10.0, key="qty")
        price = c2.number_input("ราคาต่อหุ้น", min_value=0.0, value=24.00 if market=="TH" else 100.0, step=0.01, key="price")
        tdate = c3.date_input("วันที่ซื้อ", value=date.today(), key="tdate")
        if st.button("เพิ่มดีล"):
            if symbol.strip():
                add_trade(st.session_state.user_id, symbol.strip().upper(), market, qty, price, tdate.isoformat())
                st.success("บันทึกแล้ว ✅")
                st.cache_data.clear()
            else:
                st.error("กรุณาใส่สัญลักษณ์หุ้น")

        my_trades = load_trades(st.session_state.user_id, symbol.strip().upper())
        if not my_trades.empty:
            st.write("ประวัติการซื้อขาย (สัญลักษณ์นี้)")
            st.dataframe(my_trades, use_container_width=True, hide_index=True)
            del_id = st.text_input("ลบดีล (ใส่ id)", value="", key="del_id")
            if st.button("ลบรายการตาม id"):
                try:
                    tid = int(del_id)
                    delete_trade(st.session_state.user_id, tid)
                    st.success(f"ลบดีล id={tid} แล้ว")
                    st.cache_data.clear()
                except:
                    st.error("กรุณาใส่ตัวเลข id ที่ถูกต้อง")

    # -------- Tab 1: พอร์ต + AI --------
    with tab1:
        st.subheader("สรุปพอร์ตของฉัน")
        try:
            pf = portfolio_summary(st.session_state.user_id)
        except Exception as e:
            st.error("เกิดข้อผิดพลาดตอนสร้างสรุปพอร์ต")
            st.exception(e)
            st.stop()

        st.dataframe(
            pf.style.format({
                "units":"{:,.0f}","avg_cost":"{:,.2f}","last":"{:,.2f}",
                "pnl_%":"{:,.2f}","pnl_value":"{:,.2f}",
                "ttm_div_ps": "{:,.2f}","yoc_%":"{:,.2f}","ttm_div_total":"{:,.2f}"
            }),
            use_container_width=True
        )
        if not pf.empty:
            st.download_button("ดาวน์โหลด CSV", pf.to_csv(index=False).encode("utf-8"), "portfolio.csv", "text/csv")

        st.divider()
        st.subheader("AI แนะนำ (rule-based + Indicators)")
        sym_up = symbol.strip().upper()
        user_trades = load_trades(st.session_state.user_id, sym_up)
        if user_trades.empty:
            st.info("ยังไม่มีดีลของสัญลักษณ์นี้")
        else:
            units = float(user_trades["qty"].sum())
            avg_cost = float((user_trades["qty"]*user_trades["price"]).sum() / max(units,1))
            hist = get_hist(sym_up, market, period="6mo", with_hlc=True)
            last = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else get_price(sym_up, market)
            action, buy_qty, notes, ind = rule_based_advice(avg_cost, last, hist, budget, lots)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ต้นทุนเฉลี่ย", f"{avg_cost:,.2f}")
            c2.metric("ราคาปัจจุบัน", f"{last:,.2f}")
            c3.metric("ถืออยู่ (หุ้น)", f"{units:,.0f}")
            pnl_pct = (last-avg_cost)/avg_cost*100 if avg_cost>0 else 0
            c4.metric("กำไร/ขาดทุน %", f"{pnl_pct:,.2f}%")

            c5,c6 = st.columns(2)
            c5.metric("RSI14", f"{ind['RSI14']:.1f}")
            c6.metric("ATR14", f"{ind['ATR14']:.2f}")

            st.markdown(f"**Action:** `{action}`  |  **แนะนำจำนวน:** `{buy_qty}` หุ้น")
            for n in notes:
                st.write("• " + n)

            try_notify_buy(st.session_state.user_id, sym_up, market, action, last)

        st.divider()
        st.subheader("🧠 AI สรุปพอร์ต (Gemini)")
        if st.button("สร้างสรุป"):
            result = summarize_portfolio_with_gemini(pf, model_name=gemini_model)
            st.write(result)

    # -------- Tab 3: ชาร์ตราคา --------
    with tab3:
        st.subheader(f"ชาร์ต: {symbol.strip().upper()} ({market}) + EMA/RSI/ATR")
        hist = get_hist(symbol.strip().upper(), market, period="1y", with_hlc=True)
        if hist is None or hist.empty:
            st.info("ยังไม่มีข้อมูลกราฟ")
        else:
            hist = hist.copy()
            hist["SMA20"] = sma(hist["Close"], 20)
            hist["SMA50"] = sma(hist["Close"], 50)
            hist["EMA20"] = ema(hist["Close"], 20)
            hist["EMA50"] = ema(hist["Close"], 50)
            hist["RSI14"] = rsi(hist["Close"], 14)
            hist["ATR14"] = atr(hist, 14)
            st.line_chart(hist[["Close","SMA20","SMA50","EMA20","EMA50"]])
            st.line_chart(hist[["RSI14"]])
            st.line_chart(hist[["ATR14"]])

    # -------- Tab 4: ปันผล --------
    with tab4:
        st.subheader("ปันผล / Dividends")
        c1, c2 = st.columns(2)
        if c1.button("ดึงและบันทึกปันผล (ตัวที่เลือก) 5 ปีล่าสุด"):
            n = fetch_and_store_dividends(st.session_state.user_id, symbol.strip().upper(), market, years=5)
            st.success(f"ซิงก์แล้ว (แถวใหม่/ซ้ำจะไม่เพิ่ม): {n} รายการ")
        if c2.button("ดึงและบันทึกปันผล **ทุกตัวในพอร์ต** 5 ปีล่าสุด"):
            syms = portfolio_symbols(st.session_state.user_id)
            total = 0
            for s in syms:
                total += fetch_and_store_dividends(st.session_state.user_id, s['symbol'], s['market'], years=5)
            st.success(f"ซิงก์ทั้งหมดแล้ว: +{total} รายการ")

        dv = load_dividends(st.session_state.user_id, symbol.strip().upper())
        st.markdown("**รายการปันผลที่บันทึกไว้ (ล่าสุดก่อน):**")
        if dv.empty:
            st.info("ยังไม่มีข้อมูลปันผลของสัญลักษณ์นี้ — กดปุ่มซิงก์ด้านบน")
        else:
            st.dataframe(dv.style.format({"amount":"{:,.2f}"}), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🗓️ ปฏิทินปันผล (Ex-Date)")
        if not dv.empty:
            dv = dv.copy()
            dv["ex_date"] = pd.to_datetime(dv["ex_date"])
            # ทำให้ tz-naive เพื่อความเสถียร
            try:
                dv["ex_date"] = dv["ex_date"].dt.tz_convert(None)
            except Exception:
                try:
                    dv["ex_date"] = dv["ex_date"].dt.tz_localize(None)
                except Exception:
                    pass
            years = sorted(dv["ex_date"].dt.year.unique())[::-1]
            year_sel = st.selectbox("เลือกปี", years)
            dv_year = dv[dv["ex_date"].dt.year == year_sel].copy()
            dv_year["month"] = dv_year["ex_date"].dt.strftime("%b")
            cal = dv_year.sort_values("ex_date")[["symbol","ex_date","amount","month"]]
            st.dataframe(cal, use_container_width=True)

# --------------- Entrypoint ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
else:
    app_screen()