"""Interface-only startup example; it performs no model inference or discovery.

Replace this stub with your own whitelist-model agent.  Its constant guess is
not a reference solution and does not claim any fact about a formal universe.
The trusted runner supplies the console URL, session token and output folder.
"""
from pathlib import Path
import json
import os

from rift import Console


LAW = '''import numpy as np
LAW_API_VERSION = "1.0.0"
PARAMS = {}
def acceleration(t, bodies, params):
    return np.zeros((len(bodies), 2), dtype=np.float64)
'''


def main():
    console = Console()
    mission = console.mission()
    output = Path(os.environ["RIFT_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "mission.json").write_text(json.dumps(mission, ensure_ascii=False, indent=2), encoding="utf-8")
    law = output / "law.py"
    claims = output / "claims.json"
    law.write_text(LAW, encoding="utf-8")
    # Arbitrary valid-format guesses only; no hidden truth is used.
    claims.write_text(json.dumps({"dynamics_order": 2, "hidden_matter": False,
                                  "time_dependence": False}), encoding="utf-8")
    console.save_draft(law=str(law), claims=str(claims))
    console.submit(law=str(law), claims=str(claims))


if __name__ == "__main__":
    main()
