"""
Shared slowapi rate-limiter instance.

Defined here (rather than in web/app.py) so that routers can import it
without creating a circular dependency with web/app.py.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
