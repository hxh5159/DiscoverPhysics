"""Editable entry wrapper for the runnable official LLM baseline.

Keep the full participant package on Python's module path. For a submission
whose root is the participant package, prefer ``python -m shore_agent.agent``.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shore_agent.agent import main


if __name__ == "__main__":
    raise SystemExit(main())
