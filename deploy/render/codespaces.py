"""SatQuery AI — GitHub Codespaces API client.

A tiny async client for the subset of the GitHub Codespaces REST API that the
Render orchestrator needs to wake a stopped Codespace before proxying an
inference request to it.

The Codespace serves the inference contract on a PUBLIC forwarded port
(`GET /v1/health`, `GET /v1/capabilities`, `POST /v1/analyze`, `POST /v1/assets`
— see `app/space_app.py`). Its public URL is either found in the Codespace JSON
(`web_url`, rewritten for the chosen port) or derived from its name and port.

Auth: a Personal Access Token in `GITHUB_TOKEN` (env), sent as a Bearer token.
No model, no state, no DB, no auth layer — just a thin API shim.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

__all__ = [
    "CodespaceError",
    "CodespaceAuthError",
    "CodespaceNotFoundError",
    "CodespaceAPIError",
    "forwarded_url",
    "get_codespace",
    "start_codespace",
]

GITHUB_API = "https://api.github.com"
#: A short ceiling for the control-plane calls to the GitHub API itself.
_API_TIMEOUT_S = 30.0

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CodespaceError(Exception):
    """Base class for Codespaces client failures."""


class CodespaceAuthError(CodespaceError):
    """The PAT was rejected (HTTP 401).

    Either the token is missing, expired, or lacks the `codespace` scope.
    """


class CodespaceNotFoundError(CodespaceError):
    """No Codespace exists for this account with the given name (HTTP 404)."""


class CodespaceAPIError(CodespaceError):
    """Any other non-success response from the GitHub API."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def get_codespace(token: str, name: str) -> dict[str, Any]:
    """Return the JSON description of a Codespace.

    `GET https://api.github.com/user/codespaces/{name}`. The result includes
    `state` (e.g. ``"available"``, ``"Shutdown"``, ``"Starting"``) and
    `web_url`.

    Raises:
        CodespaceAuthError: on HTTP 401.
        CodespaceNotFoundError: on HTTP 404.
        CodespaceAPIError: on any other non-2xx status.
    """
    url = f"{GITHUB_API}/user/codespaces/{name}"
    async with httpx.AsyncClient(timeout=_API_TIMEOUT_S) as client:
        resp = await client.get(url, headers=_headers(token))
    if resp.status_code == 401:
        raise CodespaceAuthError(
            "GitHub rejected the token (HTTP 401). Check GITHUB_TOKEN and that "
            "it has the 'codespace' scope."
        )
    if resp.status_code == 404:
        raise CodespaceNotFoundError(
            f"No Codespace named {name!r} was found for this account (HTTP 404)."
        )
    if resp.status_code >= 400:
        raise CodespaceAPIError(resp.status_code, resp.text[:500])
    return resp.json()


async def start_codespace(token: str, name: str) -> None:
    """Start a stopped Codespace.

    `POST https://api.github.com/user/codespaces/{name}/start`. GitHub returns
    ``202 Accepted`` (and may answer ``204``), both of which are treated as
    success. The actual transition to ``available`` is observed later by polling
    `/v1/health` on the forwarded port — this call only kicks it off.

    Raises:
        CodespaceAuthError: on HTTP 401.
        CodespaceNotFoundError: on HTTP 404.
        CodespaceAPIError: on any other non-success status.
    """
    url = f"{GITHUB_API}/user/codespaces/{name}/start"
    async with httpx.AsyncClient(timeout=_API_TIMEOUT_S) as client:
        resp = await client.post(url, headers=_headers(token))
    if resp.status_code == 401:
        raise CodespaceAuthError(
            "GitHub rejected the token (HTTP 401). Check GITHUB_TOKEN and that "
            "it has the 'codespace' scope."
        )
    if resp.status_code == 404:
        raise CodespaceNotFoundError(
            f"No Codespace named {name!r} was found for this account (HTTP 404)."
        )
    # 202 Accepted is the documented start response; 204 also seen in practice.
    if resp.status_code not in (200, 202, 204):
        raise CodespaceAPIError(resp.status_code, resp.text[:500])


def forwarded_url(codespace_json: dict[str, Any], port: int) -> str:
    """Derive the public forwarded-port URL for the inference server.

    ASSUMPTION — the public URL shape is
    ``https://{codespace-name}-{port}.app.github.dev``. This is GitHub's
    documented pattern for forwarded ports on Codespaces, but the exact host
    format was **not verifiable from the build environment** (no live Codespace
    to inspect). The implementation prefers ``codespace_json["web_url"]`` — which
    GitHub returns as a forwarded-port URL — and rewrites its trailing port
    segment to the configured ``port``. Only if ``web_url`` is absent does it
    fall back to constructing ``https://{name}-{port}.app.github.dev`` from the
    Codespace ``name``. Both branches produce the same host pattern, so the
    assumption is isolated here in one place.

    Args:
        codespace_json: the dict returned by :func:`get_codespace`.
        port: the inference port (the orchestrator's ``CODESPACE_PORT``).

    Returns:
        The public ``https://`` URL with no trailing slash.

    Raises:
        CodespaceError: if neither ``web_url`` nor ``name`` is available.
    """
    web_url = (codespace_json.get("web_url") or "").strip()
    if web_url:
        # web_url looks like https://<name>-<port>.app.github.dev — rewrite the
        # trailing port to the one the inference server is listening on.
        match = re.match(
            r"^(?P<scheme>https?://)(?P<host>.+?)-(?P<port>\d+)\.app\.github\.dev/?$",
            web_url.rstrip("/"),
        )
        if match:
            host = match.group("host")
            return f"{match.group('scheme')}{host}-{port}.app.github.dev"
        # Unexpected shape but still a URL — return it verbatim rather than guess.
        return web_url.rstrip("/")

    name = codespace_json.get("name")
    if name:
        return f"https://{name}-{port}.app.github.dev"

    raise CodespaceError(
        "Cannot derive a forwarded URL: the Codespace JSON has neither "
        "'web_url' nor 'name'."
    )
