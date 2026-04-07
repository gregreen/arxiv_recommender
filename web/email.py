"""
Email sending helpers for the arXiv Recommender.

Currently only used for email verification.  Requires:
  - resend_api_key in api_keys.json
  - verification.email_from and verification.app_base_url in email_config.json

This module is a no-op when EMAIL_VERIFICATION_ENABLED is False; callers
are expected to check that flag before calling these functions.
"""

import logging

import resend

from arxiv_lib.config import API_KEYS, APP_BASE_URL, VERIFICATION_EMAIL_FROM

log = logging.getLogger(__name__)


def _init_resend() -> None:
    key = API_KEYS.get("resend_api_key", "")
    if not key:
        raise RuntimeError(
            "resend_api_key is missing from api_keys.json. "
            "Cannot send email."
        )
    resend.api_key = key


def send_verification_email(to_address: str, token: str) -> None:
    """Send a verification email containing a one-time link.

    Raises RuntimeError if the Resend API key is missing or the send fails.
    """
    if not VERIFICATION_EMAIL_FROM:
        raise RuntimeError(
            "verification.email_from is not set in email_config.json. Cannot send email."
        )
    if not APP_BASE_URL:
        raise RuntimeError(
            "verification.app_base_url is not set in email_config.json. Cannot send email."
        )

    _init_resend()

    verify_url = f"{APP_BASE_URL}/verify-email?token={token}"

    try:
        resend.Emails.send({
            "from": VERIFICATION_EMAIL_FROM,
            "to": to_address,
            "subject": "Verify your arXiv Recommender account",
            "html": (
                "<p>Thank you for registering. Please click the link below to verify your "
                "email address. The link expires in 24 hours.</p>"
                f'<p><a href="{verify_url}">{verify_url}</a></p>'
                "<p>If you did not register for this service, you can ignore this email.</p>"
            ),
        })
    except Exception as exc:
        log.error("Failed to send verification email to %s: %s", to_address, exc)
        raise RuntimeError(f"Failed to send verification email: {exc}") from exc
