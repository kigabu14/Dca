from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import os
from datetime import datetime

FONT_PATH = "fonts/THSarabunNew.ttf"  # ต้องมีไฟล์นี้

def generate_pdf_report(results: list, output_file: str = "DCA_Report.pdf"):
    """
    สร้าง PDF รายงาน DCA จากผลวิเคราะห์
    :param results: list จาก evaluate_dca
    :param output_file: ชื่อไฟล์ที่จะสร้าง
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # โหลดฟอนต์ไทย
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont("THSarabun", FONT_PATH))
        font_name = "THSarabun"
    else:
        font_name = "Helvetica"  # fallback

    c = canvas.Canvas(output_file, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    line_height = 20

    c.setFont(font_name, 20)
    c.drawString(margin, height - margin, "รายงานวิเคราะห์หุ้นแบบ DCA")
    c.setFont(font_name, 14)
    c.drawString(margin, height - margin - 30, f"สร้างเมื่อ: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    y = height - margin - 60

    for r in results:
        c.setFont(font_name, 16)
        c.drawString(margin, y, f"หุ้น: {r['ticker']} | คะแนน: {r['score']}/100")
        y -= line_height

        c.setFont(font_name, 13)
        for reason in r['reasons']:
            c.drawString(margin + 20, y, f"- {reason}")
            y -= line_height
            if y < margin + 100:
                c.showPage()
                y = height - margin

    c.save()
    return output_file
