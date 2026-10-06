"""
The public API of an installed racerts: its modules, names, class members and call
signatures, and a comparison with a stored snapshot. tests/data/api_0.1.7.json is the
API of racerts 0.1.7 from PyPI, written with

    PYTHONPATH=<unpacked racerts-0.1.7 wheel> python tests/api_surface.py \
        tests/data/api_0.1.7.json
"""

import importlib
import inspect
import json
import logging
import pkgutil
import sys
import types

# Not API: typing and __future__ constants (and modules and loggers, below).
_SKIP = {"TYPE_CHECKING", "annotations"}


def _params(obj):
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        [p.name, None if p.default is p.empty else repr(p.default), p.kind.name]
        for p in signature.parameters.values()
    ]


def _describe(obj):
    if inspect.isclass(obj):
        members = {}
        for name, member in inspect.getmembers(obj):
            if name.startswith("_") and name != "__init__":
                continue
            if isinstance(inspect.getattr_static(obj, name), property):
                members[name] = "property"
            elif callable(member):
                members[name] = _params(member)
        return {"members": members}
    if callable(obj):
        return {"params": _params(obj)}
    if isinstance(obj, dict):
        return {"keys": sorted(map(str, obj))}
    if isinstance(obj, (bool, int, float, str)):
        return {"value": obj}
    return {}


def surface(package="racerts"):
    """{module: {name: description}} for every public name of the package."""
    root = importlib.import_module(package)
    modules = [package] + [
        m.name for m in pkgutil.walk_packages(root.__path__, package + ".")
    ]
    result = {}
    for module_name in sorted(modules):
        module = importlib.import_module(module_name)
        entry = {}
        for name, obj in vars(module).items():
            if name.startswith("_") or name in _SKIP:
                continue
            if isinstance(obj, (types.ModuleType, logging.Logger)):
                continue
            origin = getattr(obj, "__module__", None) or ""
            if callable(obj) and not origin.startswith(package):
                continue
            entry[name] = _describe(obj)
        result[module_name] = entry
    return result


def _signature_problems(where, old, new, allowed):
    """Parameters of old that new drops, reorders or gives another default."""
    if old is None or new is None:
        return []
    by_name = {p[0]: p for p in new}
    takes_kwargs = any(p[2] == "VAR_KEYWORD" for p in new)
    problems = []
    for name, default, _ in old:
        key = f"{where}:{name}"
        if key in allowed:
            continue
        if name not in by_name:
            if not takes_kwargs:
                problems.append(f"{key} removed")
        elif default is not None and by_name[name][1] != default:
            problems.append(f"{key} default {default} -> {by_name[name][1]}")
    for name, default, kind in new:
        if name not in {p[0] for p in old} and default is None:
            if kind not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                problems.append(f"{where}:{name} new required parameter")
    positional = [p[0] for p in old if p[2] == "POSITIONAL_OR_KEYWORD"]
    new_positional = [p[0] for p in new if p[2] == "POSITIONAL_OR_KEYWORD"]
    if new_positional[: len(positional)] != positional and not problems:
        problems.append(f"{where} positional order {positional} -> {new_positional}")
    return problems


def incompatibilities(snapshot, allowed=()):
    """
    Where the installed racerts differs from the snapshot: missing modules, names and
    class members, changed signatures, registry keys and constant values. Entries of
    allowed ("module.name", "Class.member" or "Class.member:parameter") are skipped.
    """
    allowed = set(allowed)
    problems = []
    for module_name, names in snapshot.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            problems.append(f"{module_name} not importable")
            continue
        for name, old in names.items():
            if f"{module_name}.{name}" in allowed:
                continue
            if not hasattr(module, name):
                problems.append(f"{module_name}.{name} missing")
                continue
            new = _describe(getattr(module, name))
            for key in ("keys", "value"):
                if key in old and new.get(key) != old[key]:
                    problems.append(
                        f"{module_name}.{name}: {key} {old[key]} -> {new.get(key)}"
                    )
            if "params" in old:
                new_params = new.get("params") or new.get("members", {}).get("__init__")
                where = f"{module_name}.{name}"
                problems += _signature_problems(where, old["params"], new_params, ())
            for member, old_params in old.get("members", {}).items():
                where = f"{name}.{member}"
                if where in allowed:
                    continue
                if member not in new.get("members", {}):
                    problems.append(f"{module_name}.{where} missing")
                elif old_params != "property":
                    new_params = new["members"][member]
                    problems += _signature_problems(
                        where, old_params, new_params, allowed
                    )
    return sorted(set(problems))


if __name__ == "__main__":
    with open(sys.argv[1], "w") as handle:
        json.dump(surface(), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
