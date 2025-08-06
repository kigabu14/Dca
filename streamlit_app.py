# streamlit_app.py
import streamlit as st
from get_financials import get_financials
from parameters import analyze_dca
from visualization import plot_radar_chart, plot_pie_chart

st.set_page_config(page_title="วิเคราะห์หุ้นแบบบัฟเฟตต์", layout="wide")

st.title("📈 วิเคราะห์หุ้นแบบ DCA ด้วยเกณฑ์ Warren Buffett")
st.markdown("เลือกรายชื่อหุ้นที่ต้องการวิเคราะห์ แล้วระบบจะดึงข้อมูลและวิเคราะห์ให้อัตโนมัติ")

# ✅ ตัวอย่างหุ้นเบื้องต้น
default_stocks = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN"]

tickers = st.multiselect(
    "เลือกหุ้นที่ต้องการวิเคราะห์ (สามารถเลือกได้หลายตัว):",
    default_stocks,
    default=default_stocks[:3]
)

if st.button("🚀 เริ่มวิเคราะห์"):
    col1, col2 = st.columns(2)
    for ticker in tickers:
        with st.spinner(f"📊 กำลังโหลดข้อมูลหุ้น {ticker}..."):
            try:
                financials = get_financials(ticker)
                if not financials:
                    st.error(f"❌ ไม่พบข้อมูลของ {ticker}")
                    continue
                score_data = analyze_dca(financials)

                with col1:
                    st.subheader(f"📌 {ticker} – สรุปคะแนน DCA")
                    st.plotly.graph_objects(plot_radar_chart(score_data, title=f"{ticker} - คะแนนรายเกณฑ์"), use_container_width=True)

                with col2:
                    st.subheader(f"📊 {ticker} – สัดส่วนคะแนน")
                    st.plotly.graph_objects(plot_pie_chart(score_data), use_container_width=True)

                st.divider()

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดกับหุ้น {ticker}: {e}")
