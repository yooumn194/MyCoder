"""CoreCoder end-to-end evaluation harness (black-box over the HTTP API).

  * dataset.json — 30 hand-written coding problems (bugfix/refactor/implement/
                   cross_file, easy/medium/hard) with deterministic pytest
                   verification.
  * runner.py    — executes the dataset against a running CoreCoder API:
                   POST /v1/agent/run -> poll /v1/agent/status -> run pytest.
  * scorer.py    — Pass@1 statistics + summary.json / report.md (+ optional
                   matplotlib chart).

The harness is fully independent: it never imports or modifies the corecoder
package, and talks to the agent only through HTTP (black-box).
"""

__version__ = "0.1.0"
