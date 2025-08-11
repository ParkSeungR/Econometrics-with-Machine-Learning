# Functions/__init__.py
# -*- coding: utf-8 -*-
"""
Auto-loader for Functions package:
- Loads every .py in this folder (except __init__.py).
- If a module defines __all__, re-export those names.
- Otherwise, re-export public callables (names not starting with '_').
- Name conflicts keep the first occurrence and print a note.
"""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Set

_pkg_dir = Path(__file__).parent
__all__: list[str] = []
_exported: Set[str] = set()

def _load_module_from_path(py_path: Path) -> ModuleType | None:
    alias = f"_functions_auto_{py_path.stem}"
    spec = spec_from_file_location(alias, str(py_path))
    if spec and spec.loader:
        mod = module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
        except Exception as e:
            print(f"[Functions] WARN: fail to import {py_path.name}: {e}")
    else:
        print(f"[Functions] WARN: cannot create spec for {py_path.name}")
    return None

def _reexport_from_module(mod: ModuleType):
    global __all__
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n, v in vars(mod).items()
                 if not n.startswith("_") and callable(v)]
    for n in names:
        if n in _exported:
            print(f"[Functions] NOTE: name conflict '{n}' (kept first)")
            continue
        try:
            globals()[n] = getattr(mod, n)
            __all__.append(n)
            _exported.add(n)
        except AttributeError:
            pass

for path in sorted(_pkg_dir.glob("*.py")):
    if path.name == "__init__.py":
        continue
    mod = _load_module_from_path(path)
    if mod is not None:
        _reexport_from_module(mod)

del Path, ModuleType, Set, _pkg_dir, _exported, _load_module_from_path, _reexport_from_module