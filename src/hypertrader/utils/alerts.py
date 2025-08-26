from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

import requests


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        return resp.ok
    except Exception:
        return False


def send_email(subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    to_addr = os.getenv("ALERT_EMAIL_TO")
    from_addr = os.getenv("ALERT_EMAIL_FROM", user or "")
    if not host or not to_addr:
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(host, port, timeout=5) as s:
            s.starttls()
            if user and password:
                s.login(user, password)
            s.sendmail(from_addr, [to_addr], msg.as_string())
        return True
    except Exception:
        return False


def alert(subject: str, body: Optional[str] = None) -> None:
    text = subject if body is None else f"{subject}\n\n{body}"
    sent = send_telegram(text)
    if not sent:
        send_email(subject, text)


