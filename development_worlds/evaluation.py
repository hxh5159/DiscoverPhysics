"""Public development reference generation and fixed-law trajectory scoring.

This design prototype evaluates the six public development worlds only.  It
uses the same public law loader and fixed integration path used for candidate
predictions.  Source data, clean references and component scores are public
here; that does not change the feedback policy of the official competition.

Run from the repository root with ``python -m development_worlds.evaluation``.
This local tool is not an execution-isolation service for untrusted programs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rift.law import load_law
from rift.observations import case_from_observation
from rift.prediction import rollout
from rift_numerics import DT, sample_indices

from . import catalog
from .worlds import DevelopmentSimulator


VERSION = "public-development-evaluation-1.0.0"
REFERENCE_VERSION = "public-development-reference-1.0.0"
ROOT = Path(__file__).resolve().parent


def _world(world):
    value = str(world)
    if value not in catalog.WORLDS:
        raise ValueError("Development world must be 1 through 6.")
    return value


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value):
    raw = json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_hashes():
    root = ROOT.parent
    names = ("development_worlds/worlds.py", "development_worlds/catalog.py",
             "development_worlds/evaluation.py", "rift/law.py",
             "rift/observations.py", "rift/prediction.py",
             "rift_numerics/__init__.py", "rift_numerics/rollout.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def claims_truth(world):
    """Public labels; source visibility concerns the simulated environment."""
    world = _world(world)
    return {"dynamics_order": 2, "hidden_matter": False,
            "time_dependence": world == "6"}


def prediction_score(mse, variance):
    """Mean squared Euclidean position error, continuously mapped to [0, 80]."""
    if not math.isfinite(float(mse)) or mse < 0:
        return 0.
    return float(80 / (1 + 10 * float(mse) / max(float(variance), 1e-12)))


def score_components(prediction, claims, truth, energy_used):
    """The common 80/15/5 arithmetic; energy is local diagnostic input here."""
    if not isinstance(claims, dict) or set(claims) != set(truth):
        raise ValueError("Claims must contain exactly three schema fields.")
    order = claims["dynamics_order"]
    if type(order) not in (int, float) or not math.isfinite(order) or order not in (1, 2):
        raise ValueError("Invalid dynamics_order.")
    if any(type(claims[key]) is not bool for key in ("hidden_matter", "time_dependence")):
        raise ValueError("Claims must use strict booleans.")
    energy = float(energy_used)
    if not math.isfinite(energy) or not 0 <= energy <= 100:
        raise ValueError("Energy must be in[0,100].")
    law = sum(5 for key, value in truth.items() if claims[key] == value)
    saving = 5 * (1 - energy / 100) if prediction >= 40 else 0.
    return {"prediction": float(prediction), "law": law, "saving": saving,
            "total": float(prediction + law + saving)}


def _variance(positions):
    flat = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    return float(np.mean(np.sum((flat - flat.mean(axis=0)) ** 2, axis=-1)))


def mission(world, simulator=None):
    """Only ordinary candidate initial-condition inputs; no true-law fields."""
    world = _world(world)
    simulator = simulator or DevelopmentSimulator(world)
    initial = simulator.background_initial_state()
    cfg = catalog.config(world)
    return {"world_id": world,
            "background_initial_state": _jsonable(initial),
            "source_property_target_id": "N0" if cfg["source_properties"] else None}


def _case_input(mission_data, case):
    # Group names and reference values do not enter the candidate callback.
    record = {key: copy.deepcopy(case[key]) for key in
              ("probes", "t_launch", "readout", "source_properties")}
    return case_from_observation(mission_data, record)


def _validate_cases(world, cases, mission_data):
    cfg = catalog.config(world)
    region = cfg["launch_region"]
    sample_counts = set()
    total = 0
    for case in cases:
        _case_input(mission_data, case)
        ticks = sample_indices(case["readout"])
        start = int(sample_indices([case["t_launch"]])[0])
        if (ticks[-1] % len(ticks) or not np.array_equal(
                ticks, np.arange(1, len(ticks) + 1) * (ticks[-1] // len(ticks)))):
            raise ValueError("Validation samples must use a uniform positive-offset schedule.")
        if (ticks[-1] * DT > cfg["maximum_single_trajectory_duration"] + 1e-12
                or (start + ticks[-1]) * DT > cfg["maximum_physical_time"] + 1e-12):
            raise ValueError("Validation time is outside the public world domain.")
        if not 1 <= len(ticks) <= 50 or not 1 <= len(case["probes"]) <= 8:
            raise ValueError("Validation case exceeds common sample/batch limits.")
        source = case["source_properties"]
        if set(source) != set(cfg["source_properties"]):
            raise ValueError("Validation source controls differ from the public catalogue.")
        attributes = [(source, cfg["source_properties"])]
        for probe in case["probes"]:
            radius = np.linalg.norm(np.asarray(probe["pos"]) - region["center"])
            if not region["radius_min"] - 1e-12 <= radius <= region["radius_max"] + 1e-12:
                raise ValueError("Validation launch position is outside its public domain.")
            if np.linalg.norm(probe["vel"]) > cfg["initial_speed_max"] + 1e-12:
                raise ValueError("Validation launch velocity is outside its public domain.")
            attributes.append((probe["properties"], cfg["probe_properties"]))
        for values, expected in attributes:
            if set(values) != set(expected):
                raise ValueError("Validation property keys differ from the public catalogue.")
            for key, value in values.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not expected[key]["min"] <= value <= expected[key]["max"]):
                    raise ValueError("Validation property is outside its public domain.")
        total += len(case["probes"])
        sample_counts.add(len(ticks))
    if total < 20 or len(sample_counts) != 1:
        raise ValueError("Validation requires at least 20 probes and a common sample count.")


def calibrate_world(world):
    """Generate clean references and full-trajectory step-halving checks.

    Test conditions are independent of an agent's observations.  No observation
    noise is added to either integration.  The half-step run only checks the
    fixed official step; it never replaces it as the scoring reference.
    """
    world = _world(world)
    simulator = DevelopmentSimulator(world)
    cases = catalog.validation_cases(world)
    mission_data = mission(world, simulator)
    _validate_cases(world, cases, mission_data)
    truths, convergence, ballistic = [], [], []
    for case in cases:
        probes = case["probes"]
        initial = [probe["pos"] for probe in probes]
        velocity = [probe["vel"] for probe in probes]
        kwargs = {"t_launch": case["t_launch"], "duration": case["readout"][-1],
                  "source_properties": case["source_properties"],
                  "probe_properties": [probe["properties"] for probe in probes]}
        coarse = simulator.integrate(initial, velocity, dt=DT, **kwargs)
        fine = simulator.integrate(initial, velocity, dt=DT / 2, **kwargs)
        ticks = sample_indices(case["readout"])
        cp, cv = np.asarray(coarse["positions"]), np.asarray(coarse["velocities"])
        fp, fv = np.asarray(fine["positions"])[::2], np.asarray(fine["velocities"])[::2]
        if cp.shape != fp.shape or cv.shape != fv.shape:
            raise ValueError("Step-halving references have inconsistent full-history shapes.")
        if not all(np.isfinite(array).all() for array in (cp, cv, fp, fv)):
            raise ValueError("A clean reference integration returned nonfinite state.")
        dx = float(np.max(np.linalg.norm(cp - fp, axis=-1)))
        dv = float(np.max(np.linalg.norm(cv - fv, axis=-1)))
        convergence.append({"name": case["name"], "max_position_difference": dx,
                            "max_velocity_difference": dv,
                            "sample_max_position_difference": float(np.max(np.linalg.norm(cp[ticks] - fp[ticks], axis=-1))),
                            "passed": dx < .001 and dv < .01})
        start = int(sample_indices([case["t_launch"]])[0])
        truths.append({"name": case["name"], "times": ((start + ticks) * DT).tolist(),
                       "positions": cp[ticks].tolist(), "velocities": cv[ticks].tolist()})
        ballistic.append(np.asarray(initial)[None] + np.asarray(case["readout"])[:, None, None]
                         * np.asarray(velocity)[None])
    all_truth = np.concatenate([np.asarray(item["positions"]).reshape(-1, 2) for item in truths])
    variance = _variance(all_truth)
    straight = np.concatenate([array.reshape(-1, 2) for array in ballistic])
    baseline_mse = float(np.mean(np.sum((straight - all_truth) ** 2, axis=-1)))
    result = {"reference_version": REFERENCE_VERSION, "evaluation_version": VERSION,
              "status": "design_prototype_not_frozen", "world": world,
              "config": catalog.config(world), "source_sha256": _source_hashes(),
              "mission": mission_data, "cases": cases, "truths": truths,
              "N_probes": sum(len(case["probes"]) for case in cases),
              "T_samples": len(cases[0]["readout"]), "global_V": variance,
              "convergence": convergence, "all_cases_converged": all(item["passed"] for item in convergence),
              "ballistic_baseline": {"MSE": baseline_mse,
                                     "e": baseline_mse / max(variance, 1e-12),
                                     "P": prediction_score(baseline_mse, variance)}}
    result["reference_sha256"] = _digest(result)
    return result


def _validated_reference(world, calibration):
    value = copy.deepcopy(calibration)
    if not isinstance(value, dict):
        raise ValueError("Calibration must be a public development reference object.")
    declared_digest = value.pop("reference_sha256", None)
    if not isinstance(declared_digest, str) or _digest(value) != declared_digest:
        raise ValueError("Reference checksum mismatch.")
    value["reference_sha256"] = declared_digest
    if (value.get("world") != world or value.get("reference_version") != REFERENCE_VERSION
            or value.get("evaluation_version") != VERSION
            or value.get("source_sha256") != _source_hashes()
            or value.get("config") != catalog.config(world)
            or value.get("cases") != catalog.validation_cases(world)):
        raise ValueError("Reference world, code or catalogue version is stale; recalibrate.")
    expected_mission = mission(world)
    if value.get("mission") != expected_mission:
        raise ValueError("Reference candidate initial-state metadata is inconsistent.")
    cases, truths = value["cases"], value.get("truths", [])
    _validate_cases(world, cases, expected_mission)
    if len(cases) != len(truths):
        raise ValueError("Reference/case counts differ.")
    flattened = []
    for case, truth in zip(cases, truths):
        expected_shape = (len(case["readout"]), len(case["probes"]), 2)
        positions = np.asarray(truth["positions"], dtype=np.float64)
        velocities = np.asarray(truth["velocities"], dtype=np.float64)
        if (positions.shape != expected_shape or velocities.shape != expected_shape
                or not np.isfinite(positions).all() or not np.isfinite(velocities).all()):
            raise ValueError("Reference trajectory shape or values are invalid.")
        expected_ticks = int(sample_indices([case["t_launch"]])[0]) + sample_indices(case["readout"])
        if truth["name"] != case["name"] or not np.array_equal(sample_indices(truth["times"]), expected_ticks):
            raise ValueError("Reference time labels or case identity differ.")
        flattened.append(positions.reshape(-1, 2))
    variance = _variance(np.concatenate(flattened))
    if not math.isfinite(float(value.get("global_V", math.nan))) or not np.isclose(
            variance, value["global_V"], rtol=1e-12, atol=1e-12):
        raise ValueError("Reference variance is inconsistent with clean positions.")
    if (value.get("all_cases_converged") is not True
            or len(value.get("convergence", [])) != len(cases)
            or any(item.get("passed") is not True for item in value["convergence"])):
        raise ValueError("A reference failed numerical convergence; no candidate is scored.")
    return value


def evaluate_law(world, law_path, claims, energy_used, calibration=None):
    """Score a real law.py via the fixed public predictor, with clean truth.

    Returns public development diagnostics.  One failed candidate trajectory
    makes its whole-world prediction score zero; successful cases are not
    silently substituted or averaged to hide the failure.
    """
    world = _world(world)
    truth_labels = claims_truth(world)
    score_components(0., claims, truth_labels, energy_used)  # Validate before costly work.
    reference = _validated_reference(world, calibrate_world(world) if calibration is None else calibration)
    model = load_law(law_path)
    errors, rows, invalid = [], [], []
    for case, truth in zip(reference["cases"], reference["truths"]):
        candidate_case = _case_input(reference["mission"], case)
        expected = np.asarray(truth["positions"], dtype=np.float64)
        try:
            output = rollout(model, candidate_case)
            predicted = np.asarray(output["positions"], dtype=np.float64)
            if predicted.shape != expected.shape or not np.isfinite(predicted).all():
                raise ValueError("Invalid candidate trajectory array.")
            if not np.array_equal(sample_indices(output["times"]), sample_indices(truth["times"])):
                raise ValueError("Candidate/reference actual time labels differ.")
            with np.errstate(over="raise", invalid="raise"):
                err = np.sum((predicted - expected) ** 2, axis=-1)
            if not np.isfinite(err).all():
                raise ValueError("Nonfinite prediction error.")
            errors.append(err.ravel())
            rows.append({"name": case["name"], "group": case["group"],
                         "MSE": float(err.mean()), "sample_records": int(err.size)})
        except MemoryError:
            # Host resource failure must not become a scientific zero score.
            raise
        except Exception as exc:
            invalid.append({"name": case["name"], "error_type": type(exc).__name__,
                            "message": str(exc)[:1000]})
    mse = None if invalid else float(np.concatenate(errors).mean())
    variance = reference["global_V"]
    prediction = 0. if invalid else prediction_score(mse, variance)
    return {"world": world, "status": "development_diagnostic", "evaluation_version": VERSION,
            **score_components(prediction, claims, truth_labels, energy_used),
            "raw_MSE": mse, "global_V": variance,
            "e": None if invalid else mse / max(variance, 1e-12),
            "energy_used": float(energy_used), "claims": copy.deepcopy(claims),
            "claims_truth": truth_labels, "law_sha256": model.source_sha256,
            "reference_sha256": reference["reference_sha256"],
            "source_sha256": reference["source_sha256"],
            "N_probes": reference["N_probes"], "T_samples": reference["T_samples"],
            "invalid_cases": invalid, "case_diagnostics": rows,
            "visibility": "public development only; not an official leaderboard result"}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, choices=catalog.WORLDS)
    parser.add_argument("--law", required=True, type=Path)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--energy-used", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-output", type=Path,
                        help="Optionally save the full public clean reference used for this run.")
    args = parser.parse_args(argv)
    claims = json.loads(args.claims.read_text(encoding="utf-8-sig"))
    score_components(0., claims, claims_truth(args.world), args.energy_used)
    reference = calibrate_world(args.world)
    result = evaluate_law(args.world, args.law, claims, args.energy_used, calibration=reference)
    if args.reference_output:
        _write_json(args.reference_output, reference)
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
