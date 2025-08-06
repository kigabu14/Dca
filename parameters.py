def analyze_dca(financials: dict) -> dict:
    score = 0
    parameters =[]

    if financials is None:
        return None

    try:
        revenue = financials['revenue']
        gross_profit = financials['gross_profit']
        ebit = financials['ebit']
        net_income = financials['net_income']
        total_assets = financials['total_assets']
        current_liabilities = financials['current_liabilities']
        equity = financials['equity']
        liabilities = financials['liabilities']
        operating_cf = financials['operating_cf']
        capex = financials['capex']
    except:
        return None

    parameters = {}

    # 1. ROE (Return on Equity) > 15%
    try:
        roe = (net_income / equity) * 100
        parameters['ROE'] = {
            'value': roe,
            'score': 1 if roe >= 15 else 0,
            'desc': 'ROE มากกว่า 15%'
        }
    except:
        parameters['ROE'] = {'value': None, 'score': 0, 'desc': 'ROE มากกว่า 15%'}

    # 2. D/E Ratio < 0.5
    try:
        de = liabilities / equity
        parameters['D/E'] = {
            'value': de,
            'score': 1 if de < 0.5 else 0,
            'desc': 'D/E ต่ำกว่า 0.5'
        }
    except:
        parameters['D/E'] = {'value': None, 'score': 0, 'desc': 'D/E ต่ำกว่า 0.5'}

    # 3. Gross Margin > 40%
    try:
        gross_margin = (gross_profit / revenue) * 100
        parameters['Gross Margin'] = {
            'value': gross_margin,
            'score': 1 if gross_margin >= 40 else 0,
            'desc': 'Gross Margin มากกว่า 40%'
        }
    except:
        parameters['Gross Margin'] = {'value': None, 'score': 0, 'desc': 'Gross Margin มากกว่า 40%'}

    # 4. Operating Margin > 20%
    try:
        operating_margin = (ebit / revenue) * 100
        parameters['Operating Margin'] = {
            'value': operating_margin,
            'score': 1 if operating_margin >= 20 else 0,
            'desc': 'Operating Margin มากกว่า 20%'
        }
    except:
        parameters['Operating Margin'] = {'value': None, 'score': 0, 'desc': 'Operating Margin มากกว่า 20%'}

    # 5. Net Profit Margin > 15%
    try:
        net_margin = (net_income / revenue) * 100
        parameters['Net Margin'] = {
            'value': net_margin,
            'score': 1 if net_margin >= 15 else 0,
            'desc': 'Net Margin มากกว่า 15%'
        }
    except:
        parameters['Net Margin'] = {'value': None, 'score': 0, 'desc': 'Net Margin มากกว่า 15%'}

    # 6. Operating Cash Flow > Net Income
    try:
        ocf_score = 1 if operating_cf > net_income else 0
        parameters['Operating CF > Net Income'] = {
            'value': f"{operating_cf:.0f} vs {net_income:.0f}",
            'score': ocf_score,
            'desc': 'กระแสเงินสดดำเนินงานมากกว่ากำไรสุทธิ'
        }
    except:
        parameters['Operating CF > Net Income'] = {'value': None, 'score': 0, 'desc': 'กระแสเงินสดดำเนินงานมากกว่ากำไรสุทธิ'}

    # 7. CAPEX ต่ำ (ค่าลงทุนไม่สูงมากเมื่อเทียบกับ OCF)
    try:
        capex_ratio = abs(capex) / operating_cf
        parameters['CAPEX Ratio'] = {
            'value': capex_ratio,
            'score': 1 if capex_ratio < 0.5 else 0,
            'desc': 'CAPEX ต่ำกว่า 50% ของกระแสเงินสดดำเนินงาน'
        }
    except:
        parameters['CAPEX Ratio'] = {'value': None, 'score': 0, 'desc': 'CAPEX ต่ำกว่า 50% ของกระแสเงินสดดำเนินงาน'}

    # 8. Current Ratio > 1.5
    try:
        current_ratio = total_assets / current_liabilities
        parameters['Current Ratio'] = {
            'value': current_ratio,
            'score': 1 if current_ratio > 1.5 else 0,
            'desc': 'Current Ratio มากกว่า 1.5'
        }
    except:
        parameters['Current Ratio'] = {'value': None, 'score': 0, 'desc': 'Current Ratio มากกว่า 1.5'}

    # 9. Net Income เติบโตจากปีก่อน (อันนี้ต้องใช้หลายปี ถ้ายังไม่มี ใช้แค่ 1 ปียังไม่รองรับ)
    parameters['Net Income Growth'] = {
        'value': 'N/A',
        'score': 0,
        'desc': 'ยังไม่รองรับการดูหลายปี'
    }

    # 10. ยอดขายเติบโตจากปีก่อน
    parameters['Revenue Growth'] = {
        'value': 'N/A',
        'score': 0,
        'desc': 'ยังไม่รองรับการดูหลายปี'
    }

    # 11. Free Cash Flow > 0 (OCF - CAPEX)
    try:
        fcf = operating_cf + capex  # capex เป็นลบอยู่แล้ว
        parameters['FCF > 0'] = {
            'value': fcf,
            'score': 1 if fcf > 0 else 0,
            'desc': 'Free Cash Flow มากกว่า 0'
        }
    except:
        parameters['FCF > 0'] = {'value': None, 'score': 0, 'desc': 'Free Cash Flow มากกว่า 0'}

    return parameters
    
 
