"""CoreCoder service layer: FastAPI HTTP endpoints + pluggable session state.

Entrypoint:
    uvicorn api.server:app --reload

Backends: STATE_BACKEND=local (default, SQLite) | redis (see state_backend.py).
Zero-intrusion: no existing corecoder module is modified; the state backend is
injected into the Orchestrator via a PersistentBlackboard (dependencies.py).
"""

__version__ = "0.1.0"
