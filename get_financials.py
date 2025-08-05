import yfinance as yf

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
           'revenue': fin.loc['Total Revenue'].values[0],
            'gross_profit': fin.loc['Gross Profit'].values[0],
            'ebit': fin.loc['EBIT'].values[0],
            'total_assets': bs.loc['Total Assets'].values[0],
            'current_liabilities': bs.loc['Current Liabilities'].values[0],
            'equity': bs.loc['Total Stockholder Equity'].values[0],
            'liabilities': bs.loc['Total Liab'].values[0],
            'operating_cf': cf.loc['Total Cash From Operating Activities'].values[0],
            'capex': cf.loc['Capital Expenditures'].values[0],
            'net_income': fin.loc['Net Income'].values[0]
        }
    except Exception as e:
        print(f"❌ Missing key data for {ticker}: {e}")
        return None
