from __future__ import annotations

import hashlib
import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier


class TunerTokenVerifier(TokenVerifier):
    """Constant-time verifier for the operator-provided HTTP bearer token."""

    def __init__(self, token: str):
        super().__init__()
        self._digest = hashlib.sha256(token.encode()).digest()

    async def verify_token(self, token: str) -> AccessToken | None:
        candidate = hashlib.sha256(token.encode()).digest()
        if not hmac.compare_digest(candidate, self._digest):
            return None
        return AccessToken(
            token=token,
            client_id="tuner-mcp-client",
            scopes=["tuner"],
        )
