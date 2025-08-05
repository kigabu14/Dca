import yfinance as yf

def safe_get_value(df, row_name):
    try:
        return df.loc[row_name].values[0]
    except Exception:
        return None

def get_financials(ticker: str):
    """
    ดึงงบการเงินของหุ้นจาก yfinance
    :param ticker: ชื่อหุ้น (เช่น 'AAPL')
    :return: dict ของข้อมูลทางการเงินที่ใช้วิเคราะห์ DCA
    """
    stock = yf.Ticker(ticker)

    try:
        fin = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
    except Exception as e:
        print(f"❌ Error loading data for {ticker}: {e}")
        return None

    try:
        return {
            'ticker': ticker,
            'revenue': safe_get_value(fin, 'Total Revenue'),
            'gross_profit': safe_get_value(fin, 'Gross Profit'),
            'ebit': safe_get_value(fin, 'EBIT'),
            'total_assets': safe_get_value(bs, 'Total Assets'),
            'current_liabilities': safe_get_value(bs, 'Current Liabilities'),
            'equity': safe_get_value(bs, 'Total Stockholder Equity'),
            'liabilities': safe_get_value(bs, 'Total Liab'),
            'operating_cf': safe_get_value(cf, 'Total Cash From Operating Activities'),
            'capex': safe_get_value(cf, 'Capital Expenditures'),
            'net_income': safe_get_value(fin, 'Net Income')
        }
    except Exception as e:
        print(f"❌ Missing key data for {ticker}: {e}")
        return None