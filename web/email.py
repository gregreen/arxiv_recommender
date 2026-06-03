"""
Email sending helpers for the arXiv Recommender.

Currently only used for email verification and password resets.  Requires:
  - For the "resend" backend (default): resend_api_key in api_keys.json
  - For the "smtp" backend: smtp_server.json in the project root
  - verification.email_from, verification.app_base_url, and
    verification.backend in email_config.json

This module is a no-op when EMAIL_VERIFICATION_ENABLED is False; callers
are expected to check that flag before calling these functions.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from arxiv_lib.config import (
    API_KEYS,
    APP_BASE_URL,
    EMAIL_BACKEND,
    SMTP_CONFIG,
    VERIFICATION_EMAIL_FROM,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send_via_resend(to: str, subject: str, html: str) -> None:
    import resend  # lazy import — not required when using the smtp backend
    key = API_KEYS.get("resend_api_key", "")
    if not key:
        raise RuntimeError(
            "resend_api_key is missing from api_keys.json. "
            "Cannot send email."
        )
    resend.api_key = key
    try:
        resend.Emails.send({
            "from": VERIFICATION_EMAIL_FROM,
            "to": to,
            "subject": subject,
            "html": html,
        })
    except Exception as exc:
        raise RuntimeError(f"Resend API call failed: {exc}") from exc


def _send_via_smtp(to: str, subject: str, html: str) -> None:
    host = SMTP_CONFIG.get("host", "")
    port = int(SMTP_CONFIG.get("port", 587))
    use_starttls: bool = bool(SMTP_CONFIG.get("use_starttls", True))
    use_ssl: bool = bool(SMTP_CONFIG.get("use_ssl", False))
    authenticated: bool = bool(SMTP_CONFIG.get("authenticated", True))
    username: str = SMTP_CONFIG.get("username", "")
    password: str = SMTP_CONFIG.get("password", "")

    if not host:
        raise RuntimeError("smtp_server.json: 'host' is not set.")
    if authenticated and not username:
        raise RuntimeError(
            "smtp_server.json: 'username' is required when 'authenticated' is true."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = VERIFICATION_EMAIL_FROM
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))

    try:
        if use_ssl:
            ctx = __import__("ssl").create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=25) as smtp:
                if authenticated:
                    smtp.login(username, password)
                smtp.sendmail(VERIFICATION_EMAIL_FROM, to, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=25) as smtp:
                if use_starttls:
                    smtp.starttls()
                if authenticated:
                    smtp.login(username, password)
                smtp.sendmail(VERIFICATION_EMAIL_FROM, to, msg.as_string())
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP send failed: {exc}") from exc


def _dispatch(to: str, subject: str, html: str) -> None:
    """Send *html* email via the configured backend."""
    if EMAIL_BACKEND == "resend":
        _send_via_resend(to, subject, html)
    elif EMAIL_BACKEND == "smtp":
        _send_via_smtp(to, subject, html)
    else:
        raise RuntimeError(
            f"Unknown email backend {EMAIL_BACKEND!r} in email_config.json. "
            "Valid values are 'resend' and 'smtp'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_verification_email(to_address: str, token: str) -> None:
    """Send a verification email containing a one-time link.

    Raises RuntimeError if configuration is missing or the send fails.
    """
    if not VERIFICATION_EMAIL_FROM:
        raise RuntimeError(
            "verification.email_from is not set in email_config.json. Cannot send email."
        )
    if not APP_BASE_URL:
        raise RuntimeError(
            "verification.app_base_url is not set in email_config.json. Cannot send email."
        )

    verify_url = f"{APP_BASE_URL}/verify-email?token={token}"
    html = (
        "<p>Thank you for registering. Please click the link below to verify your "
        "email address. The link expires in 24 hours.</p>"
        f'<p><a href="{verify_url}">{verify_url}</a></p>'
        "<p>If you did not register for this service, you can ignore this email.</p>"
    )
    try:
        _dispatch(to_address, "Verify your arXiv Recommender account", html)
    except Exception as exc:
        log.error("Failed to send verification email to %s: %s", to_address, exc)


def send_password_reset_email(to_address: str, token: str) -> None:
    """Send a password-reset email containing a one-time link.

    Raises RuntimeError if configuration is missing or the send fails.
    """
    if not VERIFICATION_EMAIL_FROM:
        raise RuntimeError(
            "verification.email_from is not set in email_config.json. Cannot send email."
        )
    if not APP_BASE_URL:
        raise RuntimeError(
            "verification.app_base_url is not set in email_config.json. Cannot send email."
        )

    reset_url = f"{APP_BASE_URL}/reset-password?token={token}"
    html = (
        "<p>We received a request to reset your arXiv Recommender password. "
        "Click the link below to set a new password. The link expires in 1 hour.</p>"
        f'<p><a href="{reset_url}">{reset_url}</a></p>'
        "<p>If you did not request a password reset, you can safely ignore this email. "
        "Your password has not been changed.</p>"
    )
    try:
        _dispatch(to_address, "Reset your arXiv Recommender password", html)
    except Exception as exc:
        log.error("Failed to send password reset email to %s: %s", to_address, exc)

