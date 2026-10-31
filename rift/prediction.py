"""Public candidate-law trajectory adapter; no true universe is queried.

Both local fitting and the organizer's trusted prediction worker use this
module. It advances the submitted acceleration law, never a hidden simulator.
Background and candidate hidden bodies start at the common physical epoch0;
probe states are inserted at their requested absolute launch time.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
import math
import numbers
import numpy as np

from rift_numerics import DT, NUMERICS_VERSION, state_rollout
from rift_numerics.rollout import ALIGNMENT_TOLERANCE_TICKS
from .law import ValidationError

CASE_SCHEMA_VERSION = "1.0.0"
DEFAULT_MAX_STEPS = 200_000
DEFAULT_MAX_STATE_RECORDS = 5_000_000
_CASE_FIELDS = frozenset(("schema_version", "t_launch", "readout", "background", "probes"))
_BODY_FIELDS = frozenset(("id", "pos", "vel", "properties"))


def _fail(path, message):
    raise ValidationError(path, message)


def _real(value, path):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        _fail(path, "expected a finite real scalar, not a boolean or string")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        _fail(path, "expected a finite real scalar")
    if not math.isfinite(number):
        _fail(path, "expected a finite real scalar")
    return number


def _tick(value, path):
    number = _real(value, path)
    if number < 0:
        _fail(path, "physical time must be nonnegative")
    scaled = number / DT
    if not math.isfinite(scaled) or scaled > np.iinfo(np.int64).max:
        _fail(path, "physical time is outside the supported integer tick range")
    tick = round(scaled)
    if abs(scaled-tick) > ALIGNMENT_TOLERANCE_TICKS:
        _fail(path, "time must align to the0.005 grid; non-grid times are not rounded")
    return tick


def _vector(value, path):
    try:
        valid_shape = np.shape(value) == (2,)
    except (ValueError, TypeError):
        valid_shape = False
    if not isinstance(value, (list, tuple, np.ndarray)) or not valid_shape:
        _fail(path, "expected a two-component real vector")
    return [_real(value[i], f"{path}[{i}]") for i in range(2)]


def _properties(value, path):
    if not isinstance(value, Mapping):
        _fail(path, "expected an object mapping property names to finite scalars")
    result = {}
    for key, number in value.items():
        if not isinstance(key, str) or not key or len(key) > 128 or not key.isprintable():
            _fail(path, "property names must be nonempty printable strings of at most128 characters")
        result[key] = _real(number, f"{path}.{key}")
    return result


def _body(value, path):
    if not isinstance(value, Mapping) or set(value) != _BODY_FIELDS:
        _fail(path, "body requires exactly id, pos, vel and properties")
    identifier = value["id"]
    if (not isinstance(identifier, str) or not identifier or len(identifier) > 128
            or not identifier.isprintable()):
        _fail(path+".id", "expected a nonempty printable stable ID of at most128 characters")
    return {"id": identifier, "pos": _vector(value["pos"], path+".pos"),
            "vel": _vector(value["vel"], path+".vel"),
            "properties": _properties(value["properties"], path+".properties")}


def validate_case(case):
    """Validate public prediction inputs and return a detached canonical dict.

    This function never executes a submitted law. Call it before candidate
    execution so an invalid organizer case is not scored as a team failure.
    Resource limits, legal exploration regions and instrument quotas are
    separate from this common mathematical input contract.
    """
    if not isinstance(case, Mapping):
        _fail("$.case", "expected a case object")
    if set(case) != _CASE_FIELDS:
        unexpected = sorted(str(key) for key in set(case)-_CASE_FIELDS)
        missing = sorted(_CASE_FIELDS-set(case))
        _fail("$.case", f"expected only the public case fields; missing={missing}, unexpected={unexpected}")
    if case["schema_version"] != CASE_SCHEMA_VERSION:
        _fail("$.case.schema_version", "expected1.0.0")
    start_tick = _tick(case["t_launch"], "$.case.t_launch")
    readout = case["readout"]
    if not isinstance(readout, (list, tuple, np.ndarray)) or np.ndim(readout) != 1 or not len(readout):
        _fail("$.case.readout", "expected a nonempty one-dimensional sequence")
    ticks = [_tick(value, f"$.case.readout[{i}]") for i, value in enumerate(readout)]
    if any(a >= b for a, b in zip(ticks, ticks[1:])):
        _fail("$.case.readout", "offsets must be strictly increasing;0 is allowed")
    if start_tick+ticks[-1] > np.iinfo(np.int64).max:
        _fail("$.case.readout", "absolute sample time exceeds supported integer ticks")
    normalized = {"schema_version": CASE_SCHEMA_VERSION,
                  "t_launch": start_tick*DT,
                  "readout": [tick*DT for tick in ticks]}
    identifiers = set()
    for name in ("background", "probes"):
        values = case[name]
        if not isinstance(values, (list, tuple)):
            _fail("$.case."+name, "expected an array of bodies")
        if name == "probes" and not values:
            _fail("$.case.probes", "at least one probe is required")
        normalized[name] = []
        for i, value in enumerate(values):
            item = _body(value, f"$.case.{name}[{i}]")
            if item["id"] in identifiers:
                _fail(f"$.case.{name}[{i}].id", "ID is duplicated within this case")
            identifiers.add(item["id"])
            normalized[name].append(item)
    return normalized


def _law_snapshot(law, case):
    """Copy and freeze model data independently of the submitted module."""
    for name in ("params", "body_props", "hidden", "acceleration"):
        if not hasattr(law, name):
            _fail("$.law", f"loaded law is missing {name}")
    if not callable(law.acceleration):
        _fail("$.law.acceleration", "expected a callable acceleration(t,bodies,params)")
    params = _properties(law.params, "$.law.PARAMS")
    if len(params) > 5:
        _fail("$.law.PARAMS", "at most five finite scalar parameters are allowed")
    if not isinstance(law.body_props, Mapping):
        _fail("$.law.BODY_PROPS", "expected an object keyed by visible background ID")
    visible = {body["id"] for body in case["background"]}
    for key in law.body_props:
        if key not in visible:
            _fail(f"$.law.BODY_PROPS.{key}", "ID is not a visible background object in this case")
    background = []
    for i, body in enumerate(case["background"]):
        properties = dict(body["properties"])
        extra = _properties(law.body_props.get(body["id"], {}), f"$.law.BODY_PROPS.{body['id']}")
        for key, value in extra.items():
            if key in properties and value != properties[key]:
                _fail(f"$.law.BODY_PROPS.{body['id']}.{key}", "cannot replace an official known input property")
            if key not in properties:
                properties[key] = value
        background.append({**body, "kind": "background", "properties": properties})
    if not isinstance(law.hidden, (list, tuple)):
        _fail("$.law.HIDDEN", "expected an array of candidate hidden bodies")
    occupied = {body["id"] for name in ("background", "probes") for body in case[name]}
    hidden = []
    for i, value in enumerate(law.hidden):
        item = _body(value, f"$.law.HIDDEN[{i}]")
        if item["id"] in occupied:
            _fail(f"$.law.HIDDEN[{i}].id", "ID duplicates a known or candidate body")
        occupied.add(item["id"])
        hidden.append({**item, "kind": "hidden"})
    probes = [{**body, "kind": "probe"} for body in case["probes"]]
    return background+hidden, probes, MappingProxyType(params)


def _readonly_vector(value):
    # Detached immutable bytes keep the candidate from modifying the numerical
    # state, even by resetting NumPy's WRITEABLE flag on the exposed view.
    return np.frombuffer(np.asarray(value, dtype=np.float64).tobytes(), dtype=np.float64)


def _bodies(metadata, positions, velocities):
    return tuple(MappingProxyType({
        "id": body["id"], "kind": body["kind"],
        "pos": _readonly_vector(positions[i]), "vel": _readonly_vector(velocities[i]),
        "properties": MappingProxyType(dict(body["properties"]))})
        for i, body in enumerate(metadata))


def _contains_boolean(value):
    # Inspect Python sequences before NumPy promotes a mixed bool/float list
    # to a floating dtype and thereby loses the original boundary type.
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind == "O":
            return any(_contains_boolean(item) for item in value.flat)
    return False


def _evaluate(law, metadata, positions, velocities, time, params):
    try:
        returned = law.acceleration(float(time), _bodies(metadata, positions, velocities), params)
        if _contains_boolean(returned):
            _fail("$.law.acceleration.return", "expected real numeric values; booleans are invalid, including inside mixed sequences")
        raw = np.asarray(returned)
    except ValidationError:
        raise
    except Exception as exc:
        _fail("$.law.acceleration", f"callback failed with {type(exc).__name__}: {str(exc)[:240]}")
    if raw.shape != (len(metadata), 2):
        _fail("$.law.acceleration.return", f"expected shape({len(metadata)},2), received{raw.shape}")
    if raw.dtype.kind not in "fiu":
        _fail("$.law.acceleration.return", "expected real numeric values; booleans, complex values and strings are invalid")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        _fail("$.law.acceleration.return", "accelerations must all be finite")
    return result


def rollout(law, case, *, max_steps=DEFAULT_MAX_STEPS,
            max_state_records=DEFAULT_MAX_STATE_RECORDS):
    """Generate candidate trajectories with the frozen organizer algorithm.

    ``max_steps`` and ``max_state_records`` are configurable resource guards
    owned by the caller/organizer, not universe parameters or instrument
    quotas. Formal evaluation supplies the same guards to every team. No
    candidate can choose the integration step or replace the numerical loop.
    """
    case = validate_case(case)
    persistent, probes, params = _law_snapshot(law, case)
    for name, value in (("max_steps", max_steps), ("max_state_records", max_state_records)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            _fail("$.limits."+name, "expected a positive integer resource guard")
    launch_tick = _tick(case["t_launch"], "$.case.t_launch")
    offsets = np.array([_tick(value, "$.case.readout") for value in case["readout"]], dtype=np.int64)
    duration_tick = int(offsets[-1])
    if launch_tick+duration_tick > max_steps:
        _fail("$.limits.max_steps", "requested trajectory exceeds the configured integration-step guard")
    count = len(persistent)
    records = (launch_tick+1)*count + (duration_tick+1)*(count+len(probes))
    if records > max_state_records:
        _fail("$.limits.max_state_records", "requested trajectory exceeds the configured state-record guard")
    pos = np.array([body["pos"] for body in persistent], dtype=np.float64).reshape(-1, 2)
    vel = np.array([body["vel"] for body in persistent], dtype=np.float64).reshape(-1, 2)
    if count:
        def background_acceleration(time, positions, velocities):
            return _evaluate(law, persistent, positions, velocities, time, params)
        history = state_rollout(background_acceleration, pos, vel,
                                t_launch=0., duration=launch_tick*DT)
        pos, vel = history["positions"][-1], history["velocities"][-1]
    pos = np.vstack((pos, np.asarray([body["pos"] for body in probes], dtype=np.float64)))
    vel = np.vstack((vel, np.asarray([body["vel"] for body in probes], dtype=np.float64)))
    indices = [np.concatenate((np.arange(count), [count+i])) for i in range(len(probes))]
    metadata = [persistent+[probe] for probe in probes]

    def acceleration(time, positions, velocities):
        result = np.empty_like(positions)
        if count:
            result[:count] = _evaluate(law, persistent, positions[:count], velocities[:count], time, params)
        for i, (selection, bodies) in enumerate(zip(indices, metadata)):
            # A probe is never present while background accelerations are
            # computed, and no probe ever sees another probe in its callback.
            result[count+i] = _evaluate(law, bodies, positions[selection], velocities[selection], time, params)[-1]
        return result

    history = state_rollout(acceleration, pos, vel, t_launch=launch_tick*DT,
                            duration=duration_tick*DT)
    return {"times": (launch_tick+offsets)*DT, "readout": offsets*DT,
            "probe_ids": [body["id"] for body in probes],
            "positions": history["positions"][offsets, count:].copy(),
            "velocities": history["velocities"][offsets, count:].copy(),
            "numerics_version": NUMERICS_VERSION}
