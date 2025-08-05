import yfinance as yf

def safe_get_value(df, key):
    try:
        return df.loc[key].iloc[0]
    except:
        return None

def get_financials(ticker: str):
    """
    ดึงข้อมูลทางการเงินของหุ้นจาก yfinance เพื่อนำไปวิเคราะห์ DCA
    """
    stock = yf.Ticker(ticker)

    try:
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
    except Exception as e:
        print(f"❌ โหลดข้อมูลไม่ได้สำหรับ {ticker}: {e}")
        return None

    try:
        return {
            'ticker': ticker,
            'revenue': safe_get_value(fin, 'Total Revenue'),                         # รายได้รวม
            'gross_profit': safe_get_value(fin, 'Gross Profit'),                     # กำไรขั้นต้น
            'ebit': safe_get_value(fin, 'EBIT'),                                     # กำไรก่อนหักภาษีและดอกเบี้ย
            'net_income': safe_get_value(fin, 'Net Income'),                         # กำไรสุทธิ
            'total_assets': safe_get_value(bs, 'Total Assets'),                      # สินทรัพย์รวม
            'current_liabilities': safe_get_value(bs, 'Current Liabilities'),        # หนี้สินหมุนเวียน
            'equity': safe_get_value(bs, 'Total Stockholder Equity'),               # ส่วนของผู้ถือหุ้น
            'liabilities': safe_get_value(bs, 'Total Liab'),                         # หนี้สินรวม
            'operating_cf': safe_get_value(cf, 'Total Cash From Operating Activities'), # กระแสเงินสดจากการดำเนินงาน
            'capex': safe_get_value(cf, 'Capital Expenditures')                      # ค่าใช้จ่ายลงทุน
        }
    except Exception as e:
        print(f"❌ ขาดข้อมูลสำคัญของ {ticker}: {e}")
        return None