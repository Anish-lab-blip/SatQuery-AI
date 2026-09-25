"""SatQuery AI — the Railway API gateway.

WHAT THIS IS
------------
A thin, stateless proxy in front of the Hugging Face Space. It is the only
component the browser talks to. `docs/DEPLOYMENT_ARCHITECTURE.md` section 2
fixes its responsibilities; this package implements them.

WHY IT EXISTS (not architectural taste)
---------------------------------------
1. The Space cannot hold the security boundary -- it is a public ASGI app.
2. GPU quota must be protected BEFORE the Space is reached. ZeroGPU gives 5
   GPU-minutes/day; a request the gateway can reject on shape must never cost
   quota.
3. Plan section 74 excludes auth/multi-tenancy/queues, so the gateway must not
   grow into those. A proxy with validation, not a platform.

WHAT IT MUST NOT DO (`docs/DEPLOYMENT_ARCHITECTURE.md` section 2.2)
------------------------------------------------------------------
No persistence. No database, no Redis, no session store. No model inference. No
auth system. No request queue. **No retries on `POST /v1/analyze`** -- a retry
spends GPU quota a second time, so the client must decide.

IMPORT-CHEAP BY CONSTRUCTION
----------------------------
This package's `__init__` imports nothing but stdlib and Pydantic-backed
modules. In particular it does NOT import FastAPI, httpx or the serving stack:
the ASGI application lives in `gateway.app` and is imported only by a server
process. That split is not stylistic -- it is what lets `gateway.policy`
(the validation, limits, CORS, request-id and error-translation logic) be
unit-tested in an environment where the ASGI framework cannot be imported.

See `gateway.policy` for that logic, and `docs/BACKEND_DEPLOYMENT_RUNBOOK.md`
section 4 for the operator configuration.

ENVIRONMENT NOTE (measured, not assumed)
----------------------------------------
In this development sandbox `import fastapi` fails with
`ImportError: DLL load failed while importing orjson ... An Application Control
policy has blocked this file` (verified as `WinError 4551` from a direct
`ctypes.CDLL` on `orjson.cp311-win_amd64.pyd`). FastAPI's own guard reads:

    try:
        orjson = importlib.import_module("orjson")
    except ModuleNotFoundError:
        orjson = None

so a policy-blocked `ImportError` escapes the guard and aborts the import. This
is an environment limitation, the same class as the documented
`pandas._libs.parsers` blocker -- NOT a defect in this package. The consequence
is recorded in `docs/PHASE19_FINAL_HARDENING.md`: `gateway.app` cannot be
imported here, so its end-to-end route behaviour is **specified and unit-tested
at the policy layer, not executed as ASGI**. The work order forbids claiming
otherwise.
"""

from __future__ import annotations

__all__: list[str] = []
