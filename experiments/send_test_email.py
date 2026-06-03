"""
Send a test email using the app's configured email backend.

Usage:
    python experiments/send_test_email.py recipient@example.com
    python experiments/send_test_email.py recipient@example.com --subject "Custom subject"
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.email import _dispatch
from arxiv_lib.config import EMAIL_BACKEND, VERIFICATION_EMAIL_FROM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test email using the app's configured email backend."
    )
    parser.add_argument("to", help="Recipient email address")
    parser.add_argument(
        "--subject",
        default="arXiv Recommender — email delivery test",
        help="Email subject line (default: %(default)s)",
    )
    args = parser.parse_args()

    html = """\
<html>
<body>
<p>Hello,</p>
<p>This is a test message from your <strong>arXiv Recommender</strong> instance
to confirm that outgoing email delivery is working correctly.</p>
<p>If you received this message, your email configuration is set up properly.</p>
<p>Configuration summary:</p>
<ul>
  <li>Backend: {backend}</li>
  <li>From address: {from_addr}</li>
</ul>
<p>No action is required — you can safely ignore this message.</p>
<p>Regards,<br>arXiv Recommender</p>
</body>
</html>
""".format(backend=EMAIL_BACKEND, from_addr=VERIFICATION_EMAIL_FROM)

    print(f"Backend  : {EMAIL_BACKEND}")
    print(f"From     : {VERIFICATION_EMAIL_FROM}")
    print(f"To       : {args.to}")
    print(f"Subject  : {args.subject}")
    print("Sending…")

    try:
        _dispatch(args.to, args.subject, html)
        print("Done — message sent successfully.")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
