import streamlit as st
from get_financials import get_financials
from evaluate_dca import calculate_dca_score
import pandas as pd

# UI config
st.set_page_config(page_title="DCA Investment Analyzer", layout="wide")
st.title("📈 DCA Investment Analyzer")
st.markdown("วิเคราะห์หุ้นต่างประเทศตามหลัก Warren Buffett + DCA พร้อมคะแนนและคำแนะนำ")

# ช่องกรอกชื่อหุ้น
tickers_input = st.text_input(
    "กรอกชื่อหุ้นที่ต้องการวิเคราะห์ (เช่น AAPL, MSFT, NVDA, SPY)", 
    value="AAPL, MSFT, NVDA"
)

if st.button("วิเคราะห์เลย 🚀"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    all_results = []

    with st.spinner("🔍 กำลังวิเคราะห์..."):
        for t in tickers:
            data = get_financials(t)
            if data:
                result = calculate_dca_score(data)
                all_results.append(result)
            else:
                st.warning(f"⚠️ โหลดข้อมูลของ {t} ไม่สำเร็จ")

    if all_results:
        # สร้างตาราง
        df = pd.DataFrame([
            {
                "Ticker": r['ticker'],
                "DCA Score": r['score'],
                "สรุป": "✅ น่าลงทุน" if r['score'] >= 70 else "❌ ยังไม่ผ่าน",
                "รายละเอียด": "\n".join(r['reasons'])
            } for r in all_results
        ])

        st.success("✅ วิเคราะห์เสร็จแล้ว")
        st.dataframe(df, use_container_width=True)

        # เตรียมไว้ส่ง PDF ต่อ
        st.session_state['dca_results'] = all_results
