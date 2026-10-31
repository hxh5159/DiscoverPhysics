"""Validate claims 4.0.0's public three-field contract using only the stdlib.

This is not a general JSON Schema engine or a scientific scorer. No candidate
law, world truth, self-consistency checker or SDK state is read. A valid format
does not establish scientific correctness.

API: parse_claims(text), load_claims(path), validate_claims(object),
canonical_json(object). Successful calls return or serialize only the three
scientific fields; contract metadata belongs to the response envelope / mission.
The two yes/no declarations require JSON booleans; numbers and strings are not
converted to booleans.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, DecimalException
import json
import math
from pathlib import Path
import sys
from typing import Any, Optional, Sequence, Union


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs/contracts/claims.schema.json"
with SCHEMA_PATH.open(encoding="utf-8-sig") as _schema_file:
    SCHEMA = json.load(_schema_file)
SCHEMA_ID = SCHEMA["$id"]
SCHEMA_VERSION = SCHEMA["x-contract-version"]
FIELD_NAMES = tuple(SCHEMA["required"])


@dataclass(frozen=True)
class ClaimsIssue:
    code: str
    pointer: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "pointer": self.pointer, "message": self.message}


class ClaimsValidationError(ValueError):
    """Public format violations, with JSON Pointer locations."""

    def __init__(self, errors: Sequence[ClaimsIssue]):
        self.errors = tuple(errors)
        super().__init__("; ".join(f"{e.code} at {e.pointer or '/'}: {e.message}" for e in self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "errors": [error.as_dict() for error in self.errors],
        }


class _ObjectPairs(list):
    """Retain pairs so duplicate keys can be reported at their full path."""


@dataclass(frozen=True)
class _NonFiniteToken:
    spelling: str


def _pointer(parent: str, key: Any) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _issue(code: str, pointer: str, message: str) -> ClaimsValidationError:
    return ClaimsValidationError([ClaimsIssue(code, pointer, message)])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _check_finite(value: Union[int, float, Decimal], pointer: str) -> None:
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise _issue("non_finite_number", pointer, "Number must fit a finite float64.") from None
    if not math.isfinite(number):
        raise _issue("non_finite_number", pointer, "Number must fit a finite float64.")


def _materialize(node: Any, pointer: str, errors: list[ClaimsIssue]) -> Any:
    if isinstance(node, _ObjectPairs):
        result = {}
        for key, value in node:
            child_pointer = _pointer(pointer, key)
            converted = _materialize(value, child_pointer, errors)
            if key in result:
                errors.append(ClaimsIssue("duplicate_key", child_pointer, "Duplicate object key is not allowed."))
            else:
                result[key] = converted
        return result
    if isinstance(node, list):
        return [_materialize(value, _pointer(pointer, index), errors) for index, value in enumerate(node)]
    if isinstance(node, _NonFiniteToken):
        errors.append(ClaimsIssue("non_finite_number", pointer, f"{node.spelling} is not a JSON number."))
        return None
    if _is_number(node):
        try:
            _check_finite(node, pointer)
        except ClaimsValidationError as error:
            errors.extend(error.errors)
    return node


def _type_matches(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return _is_number(value) and value == int(value)
    raise RuntimeError(f"Unsupported type in the claims contract: {kind}")


def _validate(value: Any, schema: dict[str, Any], pointer: str, errors: list[ClaimsIssue]) -> Any:
    kind = schema["type"]
    numeric_input = kind == "integer" and _is_number(value)
    if numeric_input:
        try:
            _check_finite(value, pointer)
        except ClaimsValidationError as error:
            errors.extend(error.errors)
            return None
    if not _type_matches(value, kind):
        errors.append(ClaimsIssue("invalid_type", pointer, f"Expected {kind}."))
        return None
    if "enum" in schema and value not in schema["enum"]:
        errors.append(ClaimsIssue("invalid_enum", pointer, "Value is not an allowed enum code."))
        return None
    if kind == "object":
        properties = schema["properties"]
        for name in schema.get("required", []):
            if name not in value:
                errors.append(ClaimsIssue("missing_field", _pointer(pointer, name), "Required field is missing."))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(ClaimsIssue("unknown_field", _pointer(pointer, name), "Field is not in this contract version."))
        return {
            name: _validate(value[name], child_schema, _pointer(pointer, name), errors)
            for name, child_schema in properties.items()
            if name in value
        }
    return int(value) if numeric_input else value


def validate_claims(value: Any) -> dict[str, Any]:
    """Return a new normalized object without mutating the submitted object.

    Required keys, strict types and enums come from the public schema.
    Every allowed combination is accepted, even if scientifically incorrect.
    """
    errors: list[ClaimsIssue] = []
    normalized = _validate(value, SCHEMA, "", errors)
    if errors:
        raise ClaimsValidationError(errors)
    return normalized


def parse_claims(text: Union[str, bytes]) -> dict[str, Any]:
    """Read strict UTF-8 JSON; a single initial UTF-8 BOM is accepted."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise _issue("invalid_encoding", "", "claims.json must be UTF-8.") from None
    elif isinstance(text, str):
        if text.startswith("\ufeff"):
            text = text[1:]
    else:
        raise _issue("invalid_input", "", "Expected JSON text or UTF-8 bytes.")
    try:
        # Keep decimal numbers exact through the integer / enum checks.
        # Otherwise 1.00000000000000000001 could round to 1 and be accepted.
        parsed = json.loads(text, object_pairs_hook=_ObjectPairs, parse_float=Decimal, parse_constant=_NonFiniteToken)
    except json.JSONDecodeError as error:
        raise _issue("invalid_json", "", f"Invalid JSON at line {error.lineno}, column {error.colno}.") from None
    except (ValueError, RecursionError, DecimalException):
        raise _issue("invalid_json", "", "JSON cannot be decoded within the parser limits.") from None
    errors: list[ClaimsIssue] = []
    try:
        materialized = _materialize(parsed, "", errors)
    except RecursionError:
        raise _issue("invalid_json", "", "JSON nesting exceeds the parser limits.") from None
    if errors:
        raise ClaimsValidationError(errors)
    return validate_claims(materialized)


def load_claims(path: Union[str, Path]) -> dict[str, Any]:
    """Load a file without modifying it; I/O failures use the same error API."""
    try:
        content = Path(path).read_bytes()
    except (OSError, ValueError) as error:
        raise _issue("io_error", "", str(error)) from None
    return parse_claims(content)


def canonical_json(value: Any) -> str:
    """Serialize only validated science fields in contract order."""
    return json.dumps(validate_claims(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _issue("usage_error", "", message)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _ArgumentParser(description="Validate claims format only; scientific correctness is not checked.")
    parser.add_argument("path", type=Path, help="UTF-8 claims.json, optionally with BOM")
    parser.add_argument("--output", type=Path, help="Write normalized claims to this path; otherwise no file is modified")
    try:
        args = parser.parse_args(argv)
        normalized = load_claims(args.path)
        result: dict[str, Any] = {
            "ok": True,
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "claims": normalized,
        }
        if args.output is not None:
            try:
                args.output.write_text(canonical_json(normalized), encoding="utf-8")
            except (OSError, ValueError) as error:
                raise _issue("io_error", "", str(error)) from None
            result["output"] = str(args.output)
        status = 0
    except ClaimsValidationError as error:
        result = error.as_dict()
        status = 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    # An invalid unknown key may contain a lone surrogate. Report it as a JSON
    # escape instead of failing while writing the diagnostic to UTF-8 stdout.
    print(serialized.encode("utf-8", errors="backslashreplace").decode("utf-8"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
