import requests

def send_telegram_message(bot_token, chat_id, message, pdf_path=None):
    """
    ส่งข้อความ (และ PDF ถ้ามี) ไปยัง Telegram
    """
    send_text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    send_doc_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    # ส่งข้อความ
    res = requests.post(send_text_url, data={
        "chat_id": chat_id,
        "text": message
    })

    # ส่ง PDF ถ้ามี
    if pdf_path and res.status_code == 200:
        with open(pdf_path, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': chat_id}
            requests.post(send_doc_url, data=data, files=files)
