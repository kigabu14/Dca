def calculate_dca_score(financial_data: dict):
    """
    ประเมินคะแนน DCA ของหุ้นหนึ่งตัว
    :param financial_data: dict ข้อมูลงบจาก get_financials()
    :return: dict รวมคะแนนและเหตุผล
    """
    try:
        score = 0
        reasons = []

        # 1. ROIC > 15%
        invested_capital = financial_data['total_assets'] - financial_data['current_liabilities']
        roic = financial_data['ebit'] / invested_capital * 100
        if roic > 15:
            score += 25
            reasons.append(f"✅ ROIC สูง ({roic:.2f}%)")
        else:
            reasons.append(f"❌ ROIC ต่ำ ({roic:.2f}%)")

        # 2. Gross Margin > 40%
        gm = financial_data['gross_profit'] / financial_data['revenue'] * 100
        if gm > 40:
            score += 15
            reasons.append(f"✅ Gross Margin สูง ({gm:.2f}%)")
        else:
            reasons.append(f"❌ Gross Margin ต่ำ ({gm:.2f}%)")

        # 3. Free Cash Flow > 0
        fcf = financial_data['operating_cf'] + financial_data['capex']  # capex เป็นลบ
        if fcf > 0:
            score += 20
            reasons.append(f"✅ FCF เป็นบวก (${fcf:,.0f})")
        else:
            reasons.append(f"❌ FCF ติดลบ (${fcf:,.0f})")

        # 4. D/E < 1
        de = financial_data['liabilities'] / financial_data['equity']
        if de < 1:
            score += 10
            reasons.append(f"✅ D/E ต่ำ ({de:.2f})")
        else:
            reasons.append(f"❌ D/E สูง ({de:.2f})")

        # 5. Revenue Growth > 0
        # NOTE: ใช้ได้แค่ปีล่าสุดเพราะเรายังไม่ดึงย้อนหลังหลายปี
        revenue = financial_data['revenue']
        net_income = financial_data['net_income']

        if revenue > 0:
            score += 20
            reasons.append(f"✅ Revenue ยังเติบโต")
        else:
            reasons.append("❌ Revenue ลดลง")

        if net_income > 0:
            score += 10
            reasons.append("✅ Net Income เป็นบวก")
        else:
            reasons.append("❌ Net Income ขาดทุน")

        return {
            'ticker': financial_data['ticker'],
            'score': score,
            'reasons': reasons,
            'summary': f"{score}/100"
        }

    except Exception as e:
        return {
            'ticker': financial_data.get('ticker', 'UNKNOWN'),
            'score': 0,
            'reasons': [f"⚠️ Error: {e}"],
            'summary': "ERROR"
        }
