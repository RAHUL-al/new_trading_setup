"""Email OTP utility — sends real OTP via Gmail SMTP."""

import smtplib
import os
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─── SMTP Config (set via environment variables) ───
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail App Password
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))


def generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return f"{random.randint(100000, 999999)}"


def send_otp_email(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send OTP to user's email. Returns True on success."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"[EMAIL] SMTP not configured. OTP for {to_email}: {otp}")
        return False

    subject = "TradePulse — Your Verification Code"

    html_body = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 32px; background: #0a0f1c; color: #e2e8f0; border-radius: 12px;">
        <div style="text-align: center; margin-bottom: 32px;">
            <span style="font-size: 36px;">📈</span>
            <h1 style="font-size: 22px; margin: 8px 0 0;">Trade<span style="color: #00ff88;">Pulse</span></h1>
        </div>
        <p style="font-size: 15px; color: #94a3b8;">Hi {user_name},</p>
        <p style="font-size: 15px; color: #94a3b8;">Your email verification code is:</p>
        <div style="text-align: center; margin: 28px 0;">
            <span style="display: inline-block; font-size: 36px; font-weight: 900; letter-spacing: 12px; padding: 16px 32px; background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,212,255,0.1)); border: 1px solid rgba(0,255,136,0.3); border-radius: 12px; color: #00ff88; font-family: 'Courier New', monospace;">
                {otp}
            </span>
        </div>
        <p style="font-size: 13px; color: #64748b; text-align: center;">This code expires in <strong style="color: #e2e8f0;">5 minutes</strong>.</p>
        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 24px 0;">
        <p style="font-size: 11px; color: #475569; text-align: center;">
            If you didn't request this code, ignore this email.<br>
            © 2026 TradePulse. Built with AngelOne SmartAPI.
        </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["From"] = f"TradePulse <{SMTP_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(f"Your TradePulse verification OTP is: {otp}\nExpires in 5 minutes.", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"[EMAIL] OTP sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL] Failed to send OTP to {to_email}: {e}")
        return False
