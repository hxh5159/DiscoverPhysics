"""One-process public candidate computation, without any simulator import."""
from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.environ.get('SHORE_SDK_ROOT', str(Path(__file__).resolve().parents[1])))

import numpy as np
from rift import load_law, rollout
from rift.observations import case_from_observation


def _replace_params(source, values):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets):
            node.value = ast.parse(repr(values), mode="eval").body
            break
    else:
        raise ValueError("PARAMS literal is required.")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def compute(request):
    source = request["source"]
    if not isinstance(source, str) or len(source.encode("utf-8")) > 1024 * 1024:
        raise ValueError("law.py must be bounded Python source text.")
    records = request.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Need at least one permitted probe trajectory.")
    # Choosing a subset is a baseline tool policy; full observations remain
    # available to the editable agent and are never replaced with hidden tests.
    records = records[-4:]
    cases = [case_from_observation(request["mission"], record) for record in records]
    expected = [np.asarray(record["positions"], dtype=float) for record in records]
    with tempfile.TemporaryDirectory(prefix="shore-candidate-") as folder:
        path = Path(folder)/"law.py"
        path.write_text(source, encoding="utf-8")
        law = load_law(path)
        names = list(law.params)
        evaluations = [0]

        def residual(values):
            parameters = {key: float(value) for key, value in zip(names, values)}
            if request.get("fit") and names:
                # Valid laws may refer to module-global PARAMS as well as the
                # callback argument. Reloading the edited source keeps both
                # representations equal throughout numerical optimization.
                path.write_text(_replace_params(source, parameters), encoding="utf-8")
                candidate = load_law(path)
            else:
                candidate = replace(law, params=parameters)
            differences = []
            for case, target in zip(cases, expected):
                prediction = np.asarray(rollout(candidate, case)["positions"])
                if prediction.shape != target.shape or not np.isfinite(prediction).all():
                    raise ValueError("Candidate produced invalid observed-data trajectories.")
                differences.append((prediction-target).reshape(-1))
            evaluations[0] += 1
            return np.concatenate(differences)

        x0 = np.asarray([law.params[key] for key in names], dtype=float)
        fitted = x0
        if request.get("fit") and names:
            from scipy.optimize import least_squares
            maximum = request.get("max_nfev", 12)
            if type(maximum) is not int or not 1 <= maximum <= 100:
                raise ValueError("max_nfev must be an integer from 1 to 100.")
            fitted = least_squares(residual, x0, max_nfev=maximum, method="trf").x
        errors = residual(fitted)
        parameters = {key: float(value) for key, value in zip(names, fitted)}
        if request.get("fit") and names:
            source = _replace_params(source, parameters)
            path.write_text(source, encoding="utf-8")
            load_law(path)
        # This is noisy observed-data MSE, never the official validation MSE.
        return {"ok": True, "law": source, "parameters": parameters,
                "observed_position_mse": float(np.mean(errors**2)*2),
                "trajectories_used": len(records), "coordinate_residuals": int(errors.size),
                "evaluations": evaluations[0], "source": "permitted observed trajectories only"}


def main():
    try:
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Candidate input too large.")
        answer = compute(json.loads(raw))
    except Exception as exc:
        answer = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
    sys.stdout.write(json.dumps(answer, ensure_ascii=False, allow_nan=False))
    return 0 if answer["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
