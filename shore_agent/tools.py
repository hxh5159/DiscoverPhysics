"""Candidate checks/fitting use observed data and the frozen public integrator."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def candidate_tool(source, mission, records, *, fit=False, timeout=30., max_nfev=12):
    """Execute one bounded local worker, never the environment simulator.

    A worker timeout is a tool failure, not an instruction to change the
    integration algorithm. This is not an operating-system security boundary.
    """
    request = {"source": source, "mission": mission, "records": records,
               "fit": bool(fit), "max_nfev": int(max_nfev)}
    worker = Path(__file__).with_name("tool_worker.py")
    try:
        result = subprocess.run([sys.executable, "-I", str(worker)],
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Candidate tool exceeded its allotted remaining time."}
    if len(result.stdout) > 2 * 1024 * 1024:
        return {"ok": False, "error": "Candidate diagnostic output was too large."}
    try:
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or type(response.get("ok")) is not bool:
            raise ValueError()
        return response
    except (ValueError, UnicodeError):
        return {"ok": False, "error": "Candidate worker did not return valid diagnostics.",
                "returncode": result.returncode}
