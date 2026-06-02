"""
Unit tests for web/email.py.

All external calls (smtplib, resend, config values) are patched so these
tests run without any real SMTP server, Resend API key, or config file on disk.
"""

import smtplib
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — patch targets
# ---------------------------------------------------------------------------

_MOD = "web.email"


def _smtp_cfg(**overrides):
    """Return a minimal valid SMTP_CONFIG dict with optional field overrides."""
    base = {
        "host": "smtp.example.com",
        "port": 587,
        "use_starttls": True,
        "use_ssl": False,
        "authenticated": True,
        "username": "user@example.com",
        "password": "secret",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _send_via_smtp
# ---------------------------------------------------------------------------

class TestSendViaSmtp:
    def _call(self, to="a@b.com", subject="Sub", html="<p>hi</p>"):
        from web.email import _send_via_smtp
        _send_via_smtp(to, subject, html)

    def test_starttls_path_calls_starttls_and_login(self):
        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        cfg = _smtp_cfg(use_starttls=True, use_ssl=False)
        with patch(f"{_MOD}.SMTP_CONFIG", cfg), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch("smtplib.SMTP", mock_smtp_cls):
            self._call()

        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("user@example.com", "secret")
        mock_smtp_instance.sendmail.assert_called_once()

    def test_ssl_path_uses_smtp_ssl_and_login(self):
        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        cfg = _smtp_cfg(use_ssl=True, use_starttls=False)
        with patch(f"{_MOD}.SMTP_CONFIG", cfg), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch("smtplib.SMTP_SSL", mock_smtp_cls), \
             patch("ssl.create_default_context", return_value=MagicMock()):
            self._call()

        mock_smtp_instance.login.assert_called_once_with("user@example.com", "secret")
        mock_smtp_instance.sendmail.assert_called_once()

    def test_unauthenticated_skips_login(self):
        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        cfg = _smtp_cfg(authenticated=False)
        with patch(f"{_MOD}.SMTP_CONFIG", cfg), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch("smtplib.SMTP", mock_smtp_cls):
            self._call()

        mock_smtp_instance.login.assert_not_called()
        mock_smtp_instance.sendmail.assert_called_once()

    def test_missing_host_raises_runtime_error(self):
        from web.email import _send_via_smtp
        with patch(f"{_MOD}.SMTP_CONFIG", _smtp_cfg(host="")), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"):
            with pytest.raises(RuntimeError, match="host"):
                _send_via_smtp("a@b.com", "Sub", "<p>hi</p>")

    def test_authenticated_without_username_raises_runtime_error(self):
        from web.email import _send_via_smtp
        with patch(f"{_MOD}.SMTP_CONFIG", _smtp_cfg(authenticated=True, username="")), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"):
            with pytest.raises(RuntimeError, match="username"):
                _send_via_smtp("a@b.com", "Sub", "<p>hi</p>")

    def test_smtp_exception_wrapped_in_runtime_error(self):
        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)
        mock_smtp_instance.sendmail.side_effect = smtplib.SMTPException("boom")

        cfg = _smtp_cfg()
        with patch(f"{_MOD}.SMTP_CONFIG", cfg), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch("smtplib.SMTP", mock_smtp_cls):
            with pytest.raises(RuntimeError, match="SMTP send failed"):
                from web.email import _send_via_smtp
                _send_via_smtp("a@b.com", "Sub", "<p>hi</p>")


# ---------------------------------------------------------------------------
# _send_via_resend
# ---------------------------------------------------------------------------

class TestSendViaResend:
    def _call(self, to="a@b.com", subject="Sub", html="<p>hi</p>"):
        from web.email import _send_via_resend
        _send_via_resend(to, subject, html)

    def test_missing_api_key_raises_runtime_error(self):
        with patch(f"{_MOD}.API_KEYS", {}), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"):
            with pytest.raises(RuntimeError, match="resend_api_key"):
                self._call()

    def test_successful_send_calls_resend(self):
        mock_resend = MagicMock()
        with patch(f"{_MOD}.API_KEYS", {"resend_api_key": "test_key"}), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch.dict("sys.modules", {"resend": mock_resend}):
            self._call()

        mock_resend.Emails.send.assert_called_once_with({
            "from": "from@example.com",
            "to": "a@b.com",
            "subject": "Sub",
            "html": "<p>hi</p>",
        })

    def test_resend_exception_wrapped_in_runtime_error(self):
        mock_resend = MagicMock()
        mock_resend.Emails.send.side_effect = Exception("api error")
        with patch(f"{_MOD}.API_KEYS", {"resend_api_key": "test_key"}), \
             patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch.dict("sys.modules", {"resend": mock_resend}):
            with pytest.raises(RuntimeError, match="Resend API call failed"):
                self._call()


# ---------------------------------------------------------------------------
# _dispatch / public API routing
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_resend_backend_routes_to_send_via_resend(self):
        with patch(f"{_MOD}.EMAIL_BACKEND", "resend"), \
             patch(f"{_MOD}._send_via_resend") as mock_resend, \
             patch(f"{_MOD}._send_via_smtp") as mock_smtp:
            from web.email import _dispatch
            _dispatch("a@b.com", "Sub", "<p>hi</p>")
        mock_resend.assert_called_once_with("a@b.com", "Sub", "<p>hi</p>")
        mock_smtp.assert_not_called()

    def test_smtp_backend_routes_to_send_via_smtp(self):
        with patch(f"{_MOD}.EMAIL_BACKEND", "smtp"), \
             patch(f"{_MOD}._send_via_resend") as mock_resend, \
             patch(f"{_MOD}._send_via_smtp") as mock_smtp:
            from web.email import _dispatch
            _dispatch("a@b.com", "Sub", "<p>hi</p>")
        mock_smtp.assert_called_once_with("a@b.com", "Sub", "<p>hi</p>")
        mock_resend.assert_not_called()

    def test_unknown_backend_raises_runtime_error(self):
        with patch(f"{_MOD}.EMAIL_BACKEND", "carrier_pigeon"):
            from web.email import _dispatch
            with pytest.raises(RuntimeError, match="carrier_pigeon"):
                _dispatch("a@b.com", "Sub", "<p>hi</p>")


# ---------------------------------------------------------------------------
# send_verification_email / send_password_reset_email — public API
# ---------------------------------------------------------------------------

class TestSendVerificationEmail:
    def test_missing_email_from_raises_before_dispatch(self):
        from web.email import send_verification_email
        with patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", ""), \
             patch(f"{_MOD}.APP_BASE_URL", "http://localhost"), \
             patch(f"{_MOD}._dispatch") as mock_dispatch:
            with pytest.raises(RuntimeError, match="email_from"):
                send_verification_email("a@b.com", "tok123")
        mock_dispatch.assert_not_called()

    def test_token_appears_in_verification_url(self):
        captured_html = {}

        def _capture(to, subject, html):
            captured_html["html"] = html

        from web.email import send_verification_email
        with patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch(f"{_MOD}.APP_BASE_URL", "http://localhost:5173"), \
             patch(f"{_MOD}._dispatch", side_effect=_capture):
            send_verification_email("a@b.com", "mytoken456")

        assert "mytoken456" in captured_html["html"]
        assert "verify-email?token=mytoken456" in captured_html["html"]


class TestSendPasswordResetEmail:
    def test_missing_email_from_raises_before_dispatch(self):
        from web.email import send_password_reset_email
        with patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", ""), \
             patch(f"{_MOD}.APP_BASE_URL", "http://localhost"), \
             patch(f"{_MOD}._dispatch") as mock_dispatch:
            with pytest.raises(RuntimeError, match="email_from"):
                send_password_reset_email("a@b.com", "tok123")
        mock_dispatch.assert_not_called()

    def test_token_appears_in_reset_url(self):
        captured_html = {}

        def _capture(to, subject, html):
            captured_html["html"] = html

        from web.email import send_password_reset_email
        with patch(f"{_MOD}.VERIFICATION_EMAIL_FROM", "from@example.com"), \
             patch(f"{_MOD}.APP_BASE_URL", "http://localhost:5173"), \
             patch(f"{_MOD}._dispatch", side_effect=_capture):
            send_password_reset_email("a@b.com", "resettoken789")

        assert "resettoken789" in captured_html["html"]
        assert "reset-password?token=resettoken789" in captured_html["html"]
