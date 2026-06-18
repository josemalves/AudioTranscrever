"""Email notifications via SMTP. Silently no-ops when SMTP is not configured."""

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587") or "587")
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip() or SMTP_USER
SMTP_TLS = os.environ.get("SMTP_TLS", "starttls").lower()  # starttls | ssl | none
APP_URL = os.environ.get("APP_URL", "http://localhost:8080").rstrip("/")

_enabled = bool(SMTP_HOST and SMTP_FROM)


def is_enabled() -> bool:
    return _enabled


def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email. Returns True on success, False otherwise."""
    if not _enabled:
        print(f"[notify] SMTP not configured — skipping email to {to}")
        return False
    if not to:
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if SMTP_TLS == "ssl":
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30) as s:
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                if SMTP_TLS == "starttls":
                    s.starttls(context=ssl.create_default_context())
                if SMTP_USER:
                    s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        return True
    except Exception as e:
        print(f"[notify] failed to send email to {to}: {e}")
        return False


def _fmt_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "—"
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def notify_job_done(to: str, username: str, filename: str, job_id: int,
                    duration_seconds: Optional[int], processing_seconds: Optional[int]) -> bool:
    subject = f"Transcrição pronta: {filename}"
    body = f"""Olá {username},

A tua transcrição "{filename}" está pronta.

Duração do áudio: {_fmt_duration(duration_seconds)}
Tempo de processamento: {_fmt_duration(processing_seconds)}

Ver e descarregar:
{APP_URL}/history/{job_id}

—
AudioTranscrever
"""
    return send_email(to, subject, body)


def notify_job_failed(to: str, username: str, filename: str, error: str) -> bool:
    subject = f"Transcrição falhou: {filename}"
    body = f"""Olá {username},

A transcrição de "{filename}" não foi concluída.

Motivo:
{error}

Podes tentar novamente em:
{APP_URL}/app

—
AudioTranscrever
"""
    return send_email(to, subject, body)


def notify_access_approved(to: str, name: str, username: str, password: str) -> bool:
    subject = "Acesso aprovado ao AudioTranscrever"
    body = f"""Olá {name},

O teu pedido de acesso foi aprovado.

URL:        {APP_URL}
Utilizador: {username}
Password:   {password}

Guarda esta password — não voltará a ser mostrada.

—
AudioTranscrever
"""
    return send_email(to, subject, body)
