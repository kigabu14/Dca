import pandas as pd
import os

def save_to_excel(ticker: str, params: dict, file_path='output/dca_result.xlsx'):
    """
    บันทึกค่าที่ได้จาก compute_parameters ลง Excel
    """
    if params is None:
        print(f"❌ ไม่สามารถบันทึก {ticker} ได้ (ไม่มีข้อมูล)")
        return

    # แปลง dict เป็น DataFrame
    df = pd.DataFrame([
        {
            'Ticker': ticker,
            'Indicator': key,
            'Value': val['value'],
            'Score': val['score'],
            'Description': val['desc']
        }
        for key, val in params.items()
    ])

    # ถ้าไฟล์ยังไม่มี ให้สร้างใหม่
    if not os.path.exists(file_path):
        df.to_excel(file_path, index=False)
    else:
        # ถ้าไฟล์มีอยู่แล้วให้ต่อท้าย
        old_df = pd.read_excel(file_path)
        combined_df = pd.concat([old_df, df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)

    print(f"✅ บันทึก {ticker} เรียบร้อยลงไฟล์: {file_path}")