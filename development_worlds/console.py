"""Noisy, budgeted local console for the six fully public development worlds.

The interaction/accounting logic follows the standard episode protocol.
Source and clean simulator data are intentionally available for development;
this object is not a service boundary for official unknown environments.
"""
from __future__ import annotations

import copy
import json
import time
from typing import Callable

import numpy as np

from .worlds import (
    DevelopmentSimulator, SINGLE_CENTER, DT, NUMERICS_VERSION,
    _finite_scalar, _tick, _sample_ticks, _state_array,
)


PROTOCOL_VERSION = "trial-console-1.1.0"


def _json_copy(value):
    """Detach payload ownership and reject JSON NaN/Inf before any exposure."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class DevelopmentConsole:
    """Public development session exposing detached JSON observations.

    config must contain a launch_region annulus, speed/time limits, explicit
    scalar property catalogs and a two-probe/eight-readout first_light request.
    All selected ranges remain calibration inputs, not official frozen rules.
    Constructor accepts no arbitrary executable candidate or path in requests.
    """

    def __init__(self, world, sigma, seed, config, *, clock: Callable = time.monotonic):
        self._sim = DevelopmentSimulator(world)
        self._sigma = _finite_scalar(sigma, "sigma")
        if self._sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        if not isinstance(config, dict):
            raise ValueError("config must be an object.")
        self._world = str(world)
        self._config = copy.deepcopy(config)
        self._clock = clock
        self._started_at = None
        self._rng = np.random.default_rng(int(seed))
        self._energy_used = 0
        self._observe_used = 0
        self._launch_calls = 0
        self._observations = []
        self._mission_payload = None
        self._transcript = []
        self._configure()

    @staticmethod
    def _positive_limit(config, name, default):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _configure(self):
        c = self._config
        self._energy_budget = self._positive_limit(c, "energy_budget", 100)
        self._observe_limit = self._positive_limit(c, "observe_calls", 20)
        self._max_probes = self._positive_limit(c, "max_probes_per_launch", 8)
        self._max_readouts = self._positive_limit(c, "max_readouts_per_probe", 50)
        self._max_observe_times = self._positive_limit(c, "max_times_per_observe", 50)
        if self._max_probes > 8 or self._max_readouts > 50 or self._max_observe_times > 50:
            raise ValueError("Calibration limits cannot exceed the current core batch/sample caps.")
        self._wall_seconds = _finite_scalar(c.get("wall_clock_seconds", 3600), "wall_clock_seconds")
        if self._wall_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive.")
        region = c.get("launch_region")
        if not isinstance(region, dict) or region.get("kind", "annulus") != "annulus":
            raise ValueError("launch_region must describe a continuous annulus.")
        center = np.asarray(region.get("center"), dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("launch_region.center must be a finite two-vector.")
        self._region_center = center.copy()
        self._r_min = _finite_scalar(region.get("radius_min"), "radius_min")
        self._r_max = _finite_scalar(region.get("radius_max"), "radius_max")
        if not 0 <= self._r_min < self._r_max:
            raise ValueError("Require 0 <= radius_min < radius_max.")
        self._speed = _finite_scalar(c.get("initial_speed_max", c.get("max_speed")), "initial_speed_max")
        if self._speed <= 0:
            raise ValueError("initial_speed_max must be positive.")
        self._time_max = _tick(c.get("maximum_physical_time", c.get("max_physical_time")), "maximum_physical_time") * DT
        self._duration_max = _tick(c.get("maximum_single_trajectory_duration", c.get("max_duration")), "maximum_single_trajectory_duration") * DT
        if not 0 < self._duration_max <= self._time_max:
            raise ValueError("Require 0 < maximum_single_trajectory_duration <= maximum_physical_time.")
        self._source_catalog = self._catalog(c.get("source_properties", {}), "source_properties")
        self._probe_catalog = self._catalog(c.get("probe_properties"), "probe_properties")
        expected_source = {"p1"} if self._world in SINGLE_CENTER else set()
        expected_probe = {"p2"} if self._world in SINGLE_CENTER else {"mass"}
        if set(self._source_catalog) != expected_source or set(self._probe_catalog) != expected_probe:
            raise ValueError("Property catalog keys do not match the configured public world inputs.")
        first = c.get("first_light")
        if not isinstance(first, dict):
            raise ValueError("config.first_light must explicitly specify the calibration first-light request.")
        if set(first) != {"probes", "t_launch", "readout", "source_properties"}:
            raise ValueError("first_light requires probes, t_launch, readout, source_properties.")
        normalized = self._validate_launch(first["probes"], first["t_launch"], first["readout"], first["source_properties"])
        probes, launch, readout, source = normalized
        if len(probes) != 2 or len(readout) != 8:
            raise ValueError("First light requires exactly two probes and eight shared sample offsets.")
        if probes[0]["properties"] != probes[1]["properties"]:
            raise ValueError("First-light probe properties must match.")
        if source != {k: item["default"] for k, item in self._source_catalog.items()}:
            raise ValueError("First light must use default source properties.")
        if probes[0]["properties"] != {k: item["default"] for k, item in self._probe_catalog.items()}:
            raise ValueError("First light must use default probe properties.")
        self._first_request = {"probes": probes, "t_launch": launch, "readout": readout, "source_properties": source}

    @staticmethod
    def _catalog(catalog, name):
        if not isinstance(catalog, dict):
            raise ValueError(f"{name} must be an object.")
        result = {}
        for key, item in catalog.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                raise ValueError(f"Invalid {name} entry.")
            lower = _finite_scalar(item.get("min", item.get("minimum")), f"{name}.{key}.min")
            upper = _finite_scalar(item.get("max", item.get("maximum")), f"{name}.{key}.max")
            default = _finite_scalar(item.get("default"), f"{name}.{key}.default")
            if not lower <= default <= upper or lower > upper:
                raise ValueError(f"{name}.{key} must have min <= default <= max.")
            result[key] = {"min": lower, "max": upper, "default": default}
        return result

    @staticmethod
    def _validate_properties(value, catalog, label):
        if value is None and not catalog:
            value = {}
        if not isinstance(value, dict) or set(value) != set(catalog):
            raise ValueError(f"{label} requires exactly {sorted(catalog)}.")
        result = {}
        for key, rule in catalog.items():
            number = _finite_scalar(value[key], f"{label}.{key}")
            if not rule["min"] <= number <= rule["max"]:
                raise ValueError(f"{label}.{key} is outside its public input range.")
            result[key] = number
        return result

    def _validate_launch(self, probes, t_launch, readout, source_properties):
        if not isinstance(probes, (list, tuple)) or not 1 <= len(probes) <= self._max_probes:
            raise ValueError(f"A launch requires 1 to {self._max_probes} probes.")
        launch_tick = _tick(t_launch, "t_launch")
        ticks = _sample_ticks(readout, "readout", limit=self._max_readouts)
        if ticks[-1] * DT > self._duration_max + 1e-12:
            raise ValueError("readout exceeds maximum_single_trajectory_duration.")
        if (launch_tick + ticks[-1]) * DT > self._time_max + 1e-12:
            raise ValueError("Absolute sample time exceeds maximum_physical_time.")
        source = self._validate_properties(source_properties, self._source_catalog, "source_properties")
        normalized = []
        for item in probes:
            if not isinstance(item, dict) or set(item) != {"pos", "vel", "properties"}:
                raise ValueError("Each probe requires exactly pos, vel and properties.")
            position = _state_array([item["pos"]], "pos")[0]
            velocity = _state_array([item["vel"]], "vel")[0]
            radius = float(np.linalg.norm(position - self._region_center))
            if not self._r_min - 1e-12 <= radius <= self._r_max + 1e-12:
                raise ValueError("Probe initial position is outside the public launch region.")
            if np.linalg.norm(velocity) > self._speed + 1e-12:
                raise ValueError("Probe initial speed exceeds the public limit.")
            props = self._validate_properties(item["properties"], self._probe_catalog, "properties")
            normalized.append({"pos": position.tolist(), "vel": velocity.tolist(), "properties": props})
        # Check physical inertia and attribute conventions before mutation.
        self._sim._properties(len(normalized), source, [p["properties"] for p in normalized])
        return normalized, launch_tick * DT, (ticks * DT).tolist(), source

    def _ensure_active(self):
        if self._started_at is None:
            raise RuntimeError("Call mission() before requesting experiments.")
        if self._clock() - self._started_at >= self._wall_seconds:
            raise TimeoutError("The exploration wall-clock window has ended.")

    def _noise_positions(self, positions):
        array = np.asarray(positions, dtype=np.float64)
        return (array + self._rng.normal(0.0, self._sigma, size=array.shape)).tolist()

    @property
    def budget(self):
        elapsed = 0.0 if self._started_at is None else max(0.0, float(self._clock() - self._started_at))
        return {"energy_remaining": self._energy_budget - self._energy_used,
                "energy_used": self._energy_used,
                "observe_remaining": self._observe_limit - self._observe_used,
                "observe_used": self._observe_used,
                "wall_clock_remaining_seconds": max(0.0, self._wall_seconds - elapsed)}

    @property
    def public_transcript(self):
        return _json_copy(self._transcript)

    def _record(self, operation, request, response):
        payload = _json_copy(response)
        self._transcript.append({"operation": operation, "request": _json_copy(request), "response": payload})
        return _json_copy(payload)

    def _first_light(self):
        request = self._first_request
        truth = self._sim.launch(**request)
        background = self._sim.background(truth["times"])
        return {**_json_copy(request), "times": truth["times"].tolist(),
                "positions": self._noise_positions(truth["positions"]),
                "velocities": truth["velocities"].tolist(),
                "visible_background_positions": self._noise_positions(background["positions"]),
                "background_ids": list(background["ids"])}

    def mission(self):
        if self._mission_payload is None:
            # Start at session creation; successful first-light computation is timed.
            self._started_at = float(self._clock())
            try:
                background = self._sim.background_initial_state()
                first = self._first_light()
                self._ensure_active()
            except Exception:
                self._started_at = None
                raise
            self._mission_payload = {
                "world_id": self._world,
                "protocol_version": PROTOCOL_VERSION,
                "numerics_version": NUMERICS_VERSION,
                "claims_schema_version": "4.0.0",
                "law_api_version": "1.0.0",
                "task": "Infer a law predicting free probe motion from permitted observations.",
                "controls": {"source_properties": self._source_catalog, "probe_properties": self._probe_catalog},
                "source_property_target_id": background["ids"][0] if self._source_catalog else None,
                "launch_region": {"kind": "annulus", "center": self._region_center.tolist(),
                                  "radius_min": self._r_min, "radius_max": self._r_max},
                "initial_speed_max": self._speed,
                "maximum_physical_time": self._time_max,
                "maximum_single_trajectory_duration": self._duration_max,
                "background_initial_state": {"ids": list(background["ids"]),
                    "positions": background["positions"].tolist(), "velocities": background["velocities"].tolist()},
                "limits": {"energy_budget": self._energy_budget, "energy_per_new_probe": 1,
                    "max_probes_per_launch": self._max_probes, "max_readouts_per_probe": self._max_readouts,
                    "observe_calls": self._observe_limit, "max_times_per_observe": self._max_observe_times,
                    "wall_clock_seconds": self._wall_seconds},
                "noise": {"positions": "independent Gaussian per coordinate on each new observation",
                          "probe_velocities": "no additional noise", "cached_data": "not re-noised"},
                "first_light": first,
            }
        self._ensure_active()
        return self._record("mission", {}, {**self._mission_payload, "budget": self.budget})

    def observe(self, times):
        self._ensure_active()
        ticks = _sample_ticks(times, "times", limit=self._max_observe_times)
        actual = ticks * DT
        if actual[-1] > self._time_max + 1e-12:
            raise ValueError("Observation exceeds maximum_physical_time.")
        if self._observe_used >= self._observe_limit:
            raise RuntimeError("The independent observe-call allowance is exhausted.")
        truth = self._sim.background(actual)
        self._ensure_active()
        positions = self._noise_positions(truth["positions"])
        self._observe_used += 1
        result = {"observation_id": f"O{self._observe_used:04d}", "times": actual.tolist(),
                  "positions": positions, "ids": list(truth["ids"]), "budget": self.budget}
        self._observations.append(_json_copy(result))
        return self._record("observe", {"times": actual.tolist()}, result)

    def launch(self, probes, t_launch, readout, source_properties=None):
        self._ensure_active()
        normalized, launch, offsets, source = self._validate_launch(probes, t_launch, readout, source_properties)
        charge = len(normalized)
        if self._energy_used + charge > self._energy_budget:
            raise RuntimeError("Insufficient energy for the requested new probes.")
        truth = self._sim.launch(normalized, launch, offsets, source)
        self._ensure_active()
        positions = self._noise_positions(truth["positions"])
        self._energy_used += charge
        self._launch_calls += 1
        request = {"probes": normalized, "t_launch": launch, "readout": offsets, "source_properties": source}
        result = {"experiment_id": f"L{self._launch_calls:04d}", **request,
                  "times": truth["times"].tolist(), "positions": positions,
                  "velocities": truth["velocities"].tolist(), "budget": self.budget}
        self._observations.append(_json_copy(result))
        return self._record("launch", request, result)

    def archive(self):
        """Read existing observations; this never runs physics or draws noise."""
        self._ensure_active()
        result = {"world_id": self._world, "first_light": self._mission_payload["first_light"],
                  "observations": self._observations, "budget": self.budget}
        return self._record("archive", {}, result)


__all__ = ["DevelopmentConsole", "PROTOCOL_VERSION"]
