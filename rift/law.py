"""Public law.py format checks and local/candidate-worker loading.

``validate_source`` and ``validate_law`` never execute submitted code. They
check the documented Python interface and literal model metadata, not physics.
``load_law`` executes the checked mathematical module and is for a team's
local environment or a separate candidate worker, never the true-world/scoring
process. AST checks and restricted builtins are not an OS isolation boundary.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable


LAW_API_VERSION = "1.0.0"
MAX_SOURCE_BYTES = 1024 * 1024  # Technical input-size guard, not a physics score.
_METADATA = frozenset(("LAW_API_VERSION", "PARAMS", "BODY_PROPS", "HIDDEN"))
_IMPORT_ROOTS = frozenset(("numpy", "scipy", "math"))
_FORBIDDEN_NAMES = frozenset((
    "open", "input", "print", "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "help", "exit", "quit", "memoryview",
))
_FORBIDDEN_ATTRIBUTES = frozenset((
    "load", "loads", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "open_memmap", "DataSource",
    "ctypes", "ctypeslib", "io", "npyio", "random", "test", "show_config",
    "f2py", "weave", "LowLevelCallable",
))
_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "pow", "range", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError",
    "IndexError", "KeyError", "OverflowError", "RuntimeError", "StopIteration",
    "TypeError", "ValueError", "ZeroDivisionError",
)


class ValidationError(ValueError):
    def __init__(self, path: str, message: str, code: str = "invalid_law"):
        self.path, self.message, self.code = path, message, code
        super().__init__(f"{path}: {message}")

    def as_dict(self):
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class LawMetadata:
    api_version: str
    params: dict
    body_props: dict
    hidden: list
    source_sha256: str
    source_name: str


@dataclass(frozen=True)
class LawModel:
    params: dict
    body_props: dict
    hidden: list
    acceleration: Callable
    api_version: str = LAW_API_VERSION
    source_sha256: str = ""
    source_name: str = "<law.py>"


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(path, "Expected a finite real scalar; bool/arrays are not scalars.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValidationError(path, "Number must fit finite float64.") from None
    if not math.isfinite(result):
        raise ValidationError(path, "Number must fit finite float64.")
    return result


def _key(value, path):
    if not isinstance(value, str) or not value or not value.isprintable() or len(value)>128:
        raise ValidationError(path, "Expected a nonempty printable string of at most128 characters.")
    return value


def _properties(value, path):
    if not isinstance(value, dict):
        raise ValidationError(path, "Expected a dictionary of finite scalar properties.")
    return {_key(k, path): _number(v, f"{path}[{k!r}]") for k, v in value.items()}


def _vector(value, path):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValidationError(path, "Expected exactly two finite coordinates.")
    return [_number(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _metadata(values):
    version = values.get("LAW_API_VERSION")
    if version != LAW_API_VERSION or type(version) is not str:
        raise ValidationError("$.LAW_API_VERSION", f"Expected {LAW_API_VERSION!r}.")
    if "PARAMS" not in values:
        raise ValidationError("$.PARAMS", "Required literal dictionary is missing.")
    params = _properties(values["PARAMS"], "$.PARAMS")
    if len(params) > 5:
        raise ValidationError("$.PARAMS", "At most five declared global fitted scalars are allowed.")
    raw = values.get("BODY_PROPS", {})
    if not isinstance(raw, dict):
        raise ValidationError("$.BODY_PROPS", "Expected an object-ID to properties dictionary.")
    body_props = {_key(k, "$.BODY_PROPS"): _properties(v, f"$.BODY_PROPS[{k!r}]") for k, v in raw.items()}
    raw = values.get("HIDDEN", [])
    if not isinstance(raw, list):
        raise ValidationError("$.HIDDEN", "Expected a literal list of candidate hidden objects.")
    hidden = []
    ids = set()
    for i, item in enumerate(raw):
        path = f"$.HIDDEN[{i}]"
        if not isinstance(item, dict) or set(item) != {"id", "pos", "vel", "properties"}:
            raise ValidationError(path, "Expected exactly id, pos, vel and properties.")
        ident = _key(item["id"], path+".id")
        if ident in ids:
            raise ValidationError(path+".id", "Duplicate hidden object ID.")
        ids.add(ident)
        hidden.append({"id": ident, "pos": _vector(item["pos"], path+".pos"),
                       "vel": _vector(item["vel"], path+".vel"),
                       "properties": _properties(item["properties"], path+".properties")})
    return version, params, body_props, hidden


def _literal(node, path):
    # ast.literal_eval silently collapses duplicate dictionary keys; reject
    # those before interpretation rather than accepting a surprising value.
    for item in ast.walk(node):
        if isinstance(item, ast.Dict):
            seen = set()
            for key in item.keys:
                try:
                    value = ast.literal_eval(key)
                except (ValueError, TypeError, SyntaxError):
                    raise ValidationError(path, "Metadata dictionary keys must be literal strings.") from None
                if not isinstance(value, str) or value in seen:
                    raise ValidationError(path, "Metadata dictionary keys must be unique literal strings.")
                seen.add(value)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        raise ValidationError(path, "Metadata must be Python literals; compute fixed mathematical constants outside the metadata dictionary.") from None


def _math_constant(node, names, module_aliases, path):
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _number(value, path)
        elif not isinstance(value, (str, bool, type(None))):
            raise ValidationError(path, "Unsupported fixed literal.")
        return value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_math_constant(v, names, module_aliases, path) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module = module_aliases.get(node.value.id)
        if module in ("math", "numpy") and node.attr in ("pi", "e", "tau"):
            return getattr(math, node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(_math_constant(node.operand, names, module_aliases, path), path)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
        left = _number(_math_constant(node.left, names, module_aliases, path), path)
        right = _number(_math_constant(node.right, names, module_aliases, path), path)
        try:
            if isinstance(node.op, ast.Add): value = left+right
            elif isinstance(node.op, ast.Sub): value = left-right
            elif isinstance(node.op, ast.Mult): value = left*right
            elif isinstance(node.op, ast.Div): value = left/right
            elif isinstance(node.op, ast.Mod): value = left % right
            else: value = left**right
        except (ArithmeticError, ValueError):
            raise ValidationError(path, "Fixed arithmetic constant is not finite real.") from None
        return _number(value, path)
    raise ValidationError(path, "Top-level fixed constants may use literals and pure arithmetic with pi/e/tau, not calls or mutable external data.")


def _check_import(node):
    path = f"$.source.line[{node.lineno}]"
    if isinstance(node, ast.Import):
        modules = [a.name for a in node.names]
    else:
        if node.level:
            raise ValidationError(path, "Relative imports are not allowed.")
        if node.module == "__future__" and all(a.name == "annotations" for a in node.names):
            return
        modules = [node.module or ""]
        modules += [(node.module or "")+"."+a.name for a in node.names]
    for module in modules:
        parts = module.split(".")
        if parts[0] not in _IMPORT_ROOTS or any(p.startswith("__") or p in _FORBIDDEN_ATTRIBUTES or p == "*" for p in parts):
            raise ValidationError(path, "Only explicit NumPy, SciPy and math scientific imports are allowed; no IO or integration-driver imports.")


def _annotation(node, path):
    if node is None:
        return
    allowed = (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.Constant,
               ast.Load, ast.BinOp, ast.BitOr)
    if any(not isinstance(item, allowed) for item in ast.walk(node)):
        raise ValidationError(path, "Annotations must be plain type expressions, not executable calls.")


def _source(source, source_name):
    if not isinstance(source, str):
        raise ValidationError("$.source", "Expected Python source text.")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValidationError("$.source", "Source exceeds the 1 MiB technical input limit.")
    try:
        tree = ast.parse(source[1:] if source.startswith('\ufeff') else source, filename="<law.py>")
    except (SyntaxError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        raise ValidationError(f"$.source.line[{line}]", "Python source does not parse.") from None
    values = {}; fixed = {}; aliases = {}; definitions = set(); acceleration = None
    for node in tree.body:
        path = f"$.source.line[{node.lineno}]"
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
            if isinstance(node, ast.Import):
                for a in node.names: aliases[a.asname or a.name.split('.')[0]] = a.name
            elif node.module in ("math", "numpy"):
                for a in node.names:
                    if a.name in ("pi", "e", "tau"): fixed[a.asname or a.name] = getattr(math,a.name)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name) or node.value is None:
                raise ValidationError(path, "Top-level declarations must assign one simple name.")
            name = targets[0].id
            if name in definitions or name.startswith("__") or name in _FORBIDDEN_NAMES:
                raise ValidationError(path, "Duplicate or reserved top-level name.")
            definitions.add(name)
            if isinstance(node, ast.AnnAssign): _annotation(node.annotation,path)
            if name in _METADATA: values[name] = _literal(node.value, "$."+name)
            else: fixed[name] = _math_constant(node.value, fixed, aliases, path)
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name in definitions or node.name in _METADATA or node.name.startswith("__"):
                raise ValidationError(path, "Duplicate or reserved function name.")
            definitions.add(node.name)
            if node.decorator_list:
                raise ValidationError(path, "Function decorators execute at import; omit them.")
            if node.name == "acceleration":
                a = node.args
                if len(a.posonlyargs)+len(a.args)!=3 or a.vararg or a.kwarg or a.kwonlyargs or a.defaults:
                    raise ValidationError("$.acceleration", "Expected acceleration(t, bodies, params) with exactly three positional arguments and no defaults.")
                acceleration = node
            for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]:
                value = _math_constant(default,fixed,aliases,path)
                if any(isinstance(item,(ast.List,ast.Dict,ast.Set)) for item in ast.walk(default)):
                    raise ValidationError(path,"Mutable helper defaults are not allowed.")
            for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]: _annotation(arg.annotation,path)
            _annotation(node.returns,path)
            continue
        raise ValidationError(path, "Top level may contain imports, literal model metadata, fixed mathematical constants and function definitions only.")
    if acceleration is None:
        raise ValidationError("$.acceleration", "Required function acceleration(t, bodies, params) is missing.")
    for node in ast.walk(tree):
        path = f"$.source.line[{getattr(node,'lineno',0)}]"
        if isinstance(node, (ast.Import,ast.ImportFrom)): _check_import(node)
        if isinstance(node, (ast.Global,ast.Nonlocal,ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom,ast.ClassDef)):
            raise ValidationError(path, "Stateful/global, async, generator and class definitions are outside this mathematical function interface.")
        if isinstance(node, ast.Name) and (node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValidationError(path, "External IO, execution/introspection and dunder names are not permitted.")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRIBUTES):
            raise ValidationError(path, "External IO, random state or introspection attributes are not permitted.")
    version, params, body_props, hidden = _metadata(values)
    result = LawMetadata(version,params,body_props,hidden,hashlib.sha256(encoded).hexdigest(),str(source_name))
    return result, tree


def validate_source(source: str, *, source_name: str = "<law.py>") -> LawMetadata:
    """Static public format inspection; does not import or execute a law."""
    return _source(source,source_name)[0]


def _read(path):
    try:
        # Preserve exact CRLF and BOM bytes for parent/worker digest checks.
        # Only parsing strips an optional leading BOM; it is part of the SHA.
        return Path(path).read_bytes().decode("utf-8")
    except (OSError,UnicodeError):
        raise ValidationError("$.source", "Cannot read UTF-8 law.py source.") from None


def validate_law(path) -> LawMetadata:
    return validate_source(_read(path),source_name=str(path))


def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)): return tuple(_freeze(v) for v in value)
    return value


def _math_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or (name.split('.')[0] not in _IMPORT_ROOTS and name != "__future__"):
        raise ImportError("Only the declared scientific imports are available.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_law(path) -> LawModel:
    """Load only in a local/team or separate candidate-worker environment.

    This function must not be called in the process holding true simulator
    objects or hidden scoring data. The import checks are not OS isolation.
    """
    source = _read(path)
    metadata, tree = _source(source,str(path))
    safe_builtins = {name:getattr(builtins,name) for name in _BUILTIN_NAMES}
    safe_builtins['__import__'] = _math_import
    namespace = {'__builtins__':safe_builtins,'__name__':'rift_candidate_law'}
    try:
        exec(compile(tree,"<law.py>","exec"),namespace,namespace)
    except Exception as exc:
        raise ValidationError("$.source", f"Mathematical module initialization failed ({type(exc).__name__}).") from None
    # Freeze candidate global metadata too, not just the params argument.
    namespace.update(PARAMS=_freeze(metadata.params),BODY_PROPS=_freeze(metadata.body_props),HIDDEN=_freeze(metadata.hidden))
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            target=node.targets[0] if isinstance(node,ast.Assign) else node.target
            namespace[target.id]=_freeze(namespace[target.id])
    fn = namespace.get('acceleration')
    if not callable(fn): raise ValidationError("$.acceleration","The acceleration entry is not callable.")
    try:
        signature = inspect.signature(fn)
        signature.bind(0.,(),MappingProxyType({}))
    except (ValueError,TypeError):
        raise ValidationError("$.acceleration","Cannot call the documented three-argument interface.") from None
    return LawModel(metadata.params,metadata.body_props,metadata.hidden,fn,metadata.api_version,
                    metadata.source_sha256,metadata.source_name)


__all__ = ['LAW_API_VERSION','MAX_SOURCE_BYTES','ValidationError','LawMetadata','LawModel',
           'validate_source','validate_law','load_law']
