
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sqlalchemy import create_engine, text
from passlib.hash import bcrypt
import re

# ---------------- App Config ----------------
st.set_page_config(page_title="Plug2Plug DCA (Login + Dropdown)", page_icon="🔐", layout="wide")
DB_URL = "sqlite:///portfolio_users.db"

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
        info = t.fast_info if hasattr(t, "fast_info") else {}
        price = float(info.get("last_price")) if info else None
    except Exception:
        price = None
    if price is None:
        hist = t.history(period="5d", interval="1d")
        price = float(hist["Close"].iloc[-1]) if len(hist) else np.nan
    return price

@st.cache_data(ttl=300)
def get_hist(symbol: str, market: str, period="6mo"):
    t = yf.Ticker(yf_symbol(symbol, market))
    df = t.history(period=period, interval="1d")
    return df

def sma(series, window):
    return series.rolling(window).mean()

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
        conn.execute(text("DELETE FROM trades WHERE id=:id AND user_id=:uid"), {"id": trade_id, "uid": user_id})

def portfolio_summary(user_id: int):
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT symbol, market, qty, price FROM trades WHERE user_id=:uid"), conn, params={"uid": user_id})
    if df.empty:
        return pd.DataFrame(columns=["symbol","market","units","avg_cost","last","pnl_%","pnl_value"])
    g = df.groupby(["symbol","market"])
    units = g["qty"].sum()
    cost = (g.apply(lambda x: (x["qty"]*x["price"]).sum()) / units).rename("avg_cost")
    out = pd.concat([units.rename("units"), cost], axis=1).reset_index()
    prices = []
    for _, row in out.iterrows():
        last = get_price(row["symbol"], row["market"])
        prices.append(last)
    out["last"] = prices
    out["pnl_%"] = (out["last"] - out["avg_cost"]) / out["avg_cost"] * 100
    out["pnl_value"] = (out["last"] - out["avg_cost"]) * out["units"]
    return out.sort_values("symbol")

def rule_based_advice(avg_cost: float, last: float, hist: pd.DataFrame, budget_month: float, lots: int):
    notes = []
    action = "HOLD"
    buy_qty = 0

    if np.isnan(last) or avg_cost <= 0 or hist is None or hist.empty:
        return action, buy_qty, ["ข้อมูลราคา/ต้นทุน/กราฟ ไม่พร้อม"]

    gap = (avg_cost - last) / avg_cost * 100  # % below avg
    hist = hist.copy()
    hist["SMA20"] = sma(hist["Close"], 20)
    hist["SMA50"] = sma(hist["Close"], 50)
    trend_up = False
    if len(hist.dropna()) > 0:
        c = hist["Close"].iloc[-1]
        s20 = hist["SMA20"].iloc[-1]
        s50 = hist["SMA50"].iloc[-1]
        trend_up = (c > s20) and (s20 > s50)

    if gap >= 5:
        action = "BUY (DCA)"
        notes.append(f"ราคาต่ำกว่าทุนเฉลี่ย ~{gap:.1f}%")
    elif trend_up:
        action = "BUY SMALL"
        notes.append("แนวโน้มขาขึ้น (Close>SMA20>SMA50) ซื้อแบ่งไม้ได้")
    else:
        action = "HOLD"
        notes.append("ยังไม่ชัด รอย่อหรือรอเบรกไฮ")

    if action.startswith("BUY"):
        per_lot_budget = max(budget_month / max(lots,1), 0)
        if last > 0:
            buy_qty = int(per_lot_budget // last)
        if buy_qty == 0 and per_lot_budget >= last*0.6:
            buy_qty = 1
        notes.append(f"งบ/ไม้ ≈ {per_lot_budget:,.0f} | แนะนำซื้อ {buy_qty} หุ้นที่ราคา ~{last:.2f}")
    notes.append("ตั้งจุดตัดขาดทุนเชิงวินัย (เช่น หลุด SMA50 หรือ -8% จากต้นทุนใหม่)")
    return action, buy_qty, notes

# --------------- Auth Layer ----------------
def register_user(username: str, password: str):
    if not valid_username(username):
        return False, "ชื่อผู้ใช้ต้องเป็น a-z A-Z 0-9 . _ - ยาว 3-32 ตัวอักษร"
    if len(password) < 6:
        return False, "รหัสผ่านต้องอย่างน้อย 6 ตัวอักษร"
    pw_hash = bcrypt.hash(password)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users(username, password_hash) VALUES(:u,:p)"),
                {"u": username, "p": pw_hash}
            )
        return True, "สมัครสมาชิกสำเร็จ"
    except Exception as e:
        msg = str(e)
        if "UNIQUE constraint failed: users.username" in msg:
            return False, "ชื่อนี้ถูกใช้แล้ว"
        return False, "สมัครไม่สำเร็จ"

def login_user(username: str, password: str):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, username, password_hash FROM users WHERE username=:u"), {"u": username}).fetchone()
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

# --------------- Ticker List (Dropdown) ----------------
def load_tickers_from(path: str, fallback: list):
    try_paths = [path, os.path.join("/mnt/data", path)]
    for p in try_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                items = sorted(df["symbol"].dropna().astype(str).unique().tolist())
                return items if items else fallback
            except Exception:
                pass
    return fallback

DEFAULT_TH = ["KTB","KBANK","SCB","PTT","PTTEP","SCC","ADVANC","AOT","CPALL","TRUE","GULF","BDMS","IVL","KCE","TOP"]
DEFAULT_US = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO","COST"]

tickers_th = load_tickers_from("tickers_th.csv", DEFAULT_TH)
tickers_us = load_tickers_from("tickers_us.csv", DEFAULT_US)

# --------------- UI ----------------
def login_screen():
    st.title("🔐 เข้าสู่ระบบ - Plug2Plug DCA")
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

    st.title("📈 Plug2Plug DCA Advisor (Per-User Portfolio)")
    st.caption("ถัวอย่างมีวินัย + คำแนะนำ rule-based แยกพอร์ตตามผู้ใช้")

    # Sidebar settings
    st.sidebar.header("📊 DCA Settings")
    market = st.sidebar.selectbox("ตลาด", ["TH","US"], key="market")

    # Dropdown with search + custom input
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

    # Tabs
    tab1, tab2, tab3 = st.tabs(["พอร์ต","บันทึกซื้อถัว","ชาร์ตราคา"])

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

        # My trades list (for current symbol)
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

    with tab1:
        st.subheader("สรุปพอร์ตของฉัน")
        pf = portfolio_summary(st.session_state.user_id)
        st.dataframe(
            pf.style.format({"units":"{:,.0f}","avg_cost":"{:,.2f}","last":"{:,.2f}",
                            "pnl_%":"{:,.2f}","pnl_value":"{:,.2f}"}),
            use_container_width=True
        )
        if not pf.empty:
            st.download_button("ดาวน์โหลด CSV", pf.to_csv(index=False).encode("utf-8"), "portfolio.csv", "text/csv")

        # Advice for current symbol if user has it
        st.divider()
        st.subheader("AI แนะนำ (rule-based)")
        sym_up = symbol.strip().upper()
        user_trades = load_trades(st.session_state.user_id, sym_up)
        if user_trades.empty:
            st.info("ยังไม่มีดีลของสัญลักษณ์นี้")
        else:
            units = user_trades["qty"].sum()
            avg_cost = (user_trades["qty"]*user_trades["price"]).sum() / max(units,1)
            last = get_price(sym_up, market)
            hist = get_hist(sym_up, market, period="6mo")
            action, buy_qty, notes = rule_based_advice(avg_cost, last, hist, budget, lots)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ต้นทุนเฉลี่ย", f"{avg_cost:,.2f}")
            c2.metric("ราคาปัจจุบัน", f"{last:,.2f}")
            c3.metric("ถืออยู่ (หุ้น)", f"{units:,.0f}")
            pnl_pct = (last-avg_cost)/avg_cost*100 if avg_cost>0 else 0
            c4.metric("กำไร/ขาดทุน %", f"{pnl_pct:,.2f}%")

            st.markdown(f"**Action:** `{action}`  |  **แนะนำจำนวน:** `{buy_qty}` หุ้น")
            for n in notes:
                st.write("• " + n)

    with tab3:
        st.subheader(f"ชาร์ต: {symbol.strip().upper()} ({market})")
        hist = get_hist(symbol.strip().upper(), market, period="1y")
        if hist is None or hist.empty:
            st.info("ยังไม่มีข้อมูลกราฟ")
        else:
            hist = hist.copy()
            hist["SMA20"] = sma(hist["Close"], 20)
            hist["SMA50"] = sma(hist["Close"], 50)
            st.line_chart(hist[["Close","SMA20","SMA50"]])

# --------------- Entrypoint ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
else:
    app_screen()
