# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Validator for ONE global target configuration packages.

What this script checks:
1) Target INI existence & minimal required keys (TARGET, BACKEND).
2) Command schema files (codegen.py, profile.py) exist for BACKEND.
3) Command schema is importable without ONE runtime by shimming `onelib.argumentparse`.
4) "Required" arguments are present in schemas:
   - codegen: DriverName, TargetOption, input{,_path}, output{,_path}
   - profile: DriverName, TargetOption, input{,_path}

Usage:
  python tools/validate_global_conf.py --root . \
      --target {TARGET_NAME} --backend {BACKEND_NAME}

You can also point to the "installed" layout:
  python tools/validate_global_conf.py --installed \
      --target {TARGET_NAME} --backend {BACKEND_NAME}
"""
import argparse
import configparser
import importlib.util
import io
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Minimal shim for onelib.argumentparse
# -----------------------------
class _Action:
    pass


class DriverName(_Action):
    name = "DriverName"


class TargetOption(_Action):
    name = "TargetOption"


class NormalOption(_Action):
    name = "NormalOption"


@dataclass
class _ArgSpec:
    names: Tuple[str, ...]
    action: type
    dtype: Optional[type] = None


class ArgumentParser:
    """
    A very small recorder that emulates the subset used by command_schema().
    """
    def __init__(self):
        self.args: List[_ArgSpec] = []

    def add_argument(self, *names: str, action=_Action, dtype: Optional[type] = None):
        if not names:
            raise ValueError("add_argument requires at least one name")
        # Normalize to strings
        names = tuple(str(n) for n in names)
        self.args.append(_ArgSpec(names=names, action=action, dtype=dtype))
        return self


# Expose shim as "onelib.argumentparse"
def _install_onelib_shim():
    mod_onelib = types.ModuleType("onelib")
    mod_argparse = types.ModuleType("onelib.argumentparse")
    mod_argparse.ArgumentParser = ArgumentParser
    mod_argparse.DriverName = DriverName
    mod_argparse.TargetOption = TargetOption
    mod_argparse.NormalOption = NormalOption

    # Both import styles are used in examples:
    #   from onelib import argumentparse
    #   from onelib.argumentparse import DriverName, ...
    sys.modules["onelib"] = mod_onelib
    sys.modules["onelib.argumentparse"] = mod_argparse
    mod_onelib.argumentparse = mod_argparse


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class SchemaReport:
    file_path: str
    ok: bool
    messages: List[str] = field(default_factory=list)


def _load_schema(file_path: str) -> Tuple[List[_ArgSpec], SchemaReport]:
    rep = SchemaReport(file_path=file_path, ok=False)
    if not os.path.isfile(file_path):
        rep.messages.append(f"Missing schema file: {file_path}")
        return [], rep

    _install_onelib_shim()
    spec = importlib.util.spec_from_file_location("schema_module", file_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        rep.messages.append(f"Import error: {e}")
        return [], rep
    if not hasattr(mod, "command_schema"):
        rep.messages.append("Schema module has no `command_schema()` function.")
        return [], rep
    try:
        parser = mod.command_schema()
    except Exception as e:
        rep.messages.append(f"command_schema() execution failed: {e}")
        return [], rep
    if not isinstance(parser, ArgumentParser):
        rep.messages.append(
            "command_schema() did not return an ArgumentParser instance (shim).")
        return [], rep

    rep.ok = True
    return parser.args, rep


def _has_action(args: List[_ArgSpec], action_type: type) -> bool:
    return any(a.action is action_type for a in args)


def _has_any_name(args: List[_ArgSpec], names: List[str]) -> bool:
    for a in args:
        for n in a.names:
            if n in names:
                return True
    return False


def _check_codegen_contract(args: List[_ArgSpec]) -> List[str]:
    errs = []
    if not _has_action(args, DriverName):
        errs.append("Missing DriverName action.")
    if not _has_action(args, TargetOption):
        errs.append("Missing TargetOption action.")
    if not _has_any_name(args, ["input", "input_path"]):
        errs.append("Missing input/input_path argument.")
    if not _has_any_name(args, ["--output", "--output_path"]):
        errs.append("Missing --output/--output_path option.")
    return errs


def _check_profile_contract(args: List[_ArgSpec]) -> Tuple[List[str], List[str]]:
    errs, warns = [], []
    if not _has_action(args, DriverName):
        errs.append("Missing DriverName action.")
    if not _has_action(args, TargetOption):
        errs.append("Missing TargetOption action.")
    if not _has_any_name(args, ["input", "input_path"]):
        errs.append("Missing input/input_path argument.")
    return errs, warns


def _read_ini(path: str) -> Dict[str, str]:
    # configparser with ; as comment
    parser = configparser.ConfigParser(strict=False,
                                       interpolation=None,
                                       delimiters=("=", ))
    # Treat keys as case-sensitive; normalize ourselves by not lowercasing
    parser.optionxform = str  # preserve case
    with io.open(path, "r", encoding="utf-8") as f:
        # Place everything in DEFAULT to keep simple key=value structure
        content = "[DEFAULT]\n" + f.read()
    parser.read_string(content)
    return dict(parser["DEFAULT"])


def _resolve_paths(root: str, target: str, backend: str, installed: bool):
    if installed:
        target_ini = f"/usr/share/one/target/{target}.ini"
        codegen_py = f"/usr/share/one/backends/command/{backend}/codegen.py"
        profile_py = f"/usr/share/one/backends/command/{backend}/profile.py"
    else:
        target_ini = os.path.join(root, "target", target, f"{target}.ini")
        codegen_py = os.path.join(root, "backend", backend, "one-cmds", "codegen.py")
        profile_py = os.path.join(root, "backend", backend, "one-cmds", "profile.py")
    return target_ini, codegen_py, profile_py


def main():
    ap = argparse.ArgumentParser(
        description="Validate ONE global target configuration package.")
    ap.add_argument("--root", default=".", help="Repo root (for source-tree validation).")
    ap.add_argument("--target", required=True, help="Target name, e.g., Rose")
    ap.add_argument("--backend", required=True, help="Backend name, e.g., dummy")
    ap.add_argument("--installed",
                    action="store_true",
                    help="Validate installed paths instead of source tree.")
    args = ap.parse_args()

    target_ini, codegen_py, profile_py = _resolve_paths(args.root, args.target,
                                                        args.backend, args.installed)

    print("== ONE Global Conf Validator ==")
    print(f"Mode      : {'installed' if args.installed else 'source-tree'}")
    print(f"Target INI: {target_ini}")
    print(f"Codegen   : {codegen_py}")
    print(f"Profile   : {profile_py}")
    print("")

    # 1) INI
    if not os.path.isfile(target_ini):
        print(f"[ERROR] Target INI not found: {target_ini}")
        return 2
    kv = _read_ini(target_ini)
    errors = []
    for k in ("TARGET", "BACKEND"):
        if k not in kv or not kv[k].strip():
            errors.append(f"Missing required key {k} in INI.")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return 2

    target_val = kv["TARGET"].strip()
    backend_val = kv["BACKEND"].strip()
    if target_val != args.target:
        print(f"[WARN] TARGET in INI is '{target_val}' but --target is '{args.target}'")
    if backend_val != args.backend:
        print(
            f"[WARN] BACKEND in INI is '{backend_val}' but --backend is '{args.backend}'")

    # 2) Load schemas
    codegen_args, codegen_rep = _load_schema(codegen_py)
    profile_args, profile_rep = _load_schema(profile_py)

    ok = True

    def print_report(name: str, rep: SchemaReport):
        nonlocal ok
        status = "OK" if rep.ok else "FAIL"
        print(f"[{name}] {rep.file_path}: {status}")
        for m in rep.messages:
            print(f"  - {m}")
        if not rep.ok:
            ok = False

    print_report("SCHEMA", codegen_rep)
    print_report("SCHEMA", profile_rep)

    # 3) Contracts
    if codegen_rep.ok:
        cerrs = _check_codegen_contract(codegen_args)
        for e in cerrs:
            print(f"[ERROR] codegen schema: {e}")
        ok = ok and not cerrs
    if profile_rep.ok:
        perrs, pwrns = _check_profile_contract(profile_args)
        for e in perrs:
            print(f"[ERROR] profile schema: {e}")
        for w in pwrns:
            print(f"[WARN] profile schema: {w}")
        ok = ok and not perrs

    print("")
    if ok:
        print("[PASS] Validation succeeded.")
        return 0
    else:
        print("[FAIL] Validation failed.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
