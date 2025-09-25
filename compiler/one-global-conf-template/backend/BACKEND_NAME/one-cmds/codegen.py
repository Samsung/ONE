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

# This file defines the "command schema" for one-codegen when the selected target
# uses BACKEND={BACKEND_NAME}. It allows onecc to accept key-value options in the
# [one-codegen] section instead of a free-form "command" string.
#
# Install path (required by ONE):
#   /usr/share/one/backends/command/{BACKEND_NAME}/codegen.py
#
# Notes:
# - You MUST expose a top-level function named `command_schema()`.
# - Use actions:
#     * DriverName   : name of the backend driver executable (positional, first)
#     * TargetOption : special option conveying the selected TARGET to the driver
#     * NormalOption : regular driver options (positional or optional)
# - If an argument should accept both ONE-style keys (e.g., input_path) and a
#   plain alias (e.g., input), pass multiple names to add_argument.
#
# Example onecc snippet (TO-BE):
#   [one-codegen]
#   output_path=out.tvn
#   input_path=model.opt.circle
#   verbose=True
#
# The resolved command (conceptually) becomes:
#   {DRIVER_NAME} --target {TARGET_NAME} --verbose --output out.tvn model.opt.circle

from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()

    # 1) Driver binary name (positional). This MUST match your actual driver.
    #    Example: "dummy-compile", "triv-compile", etc.
    parser.add_argument("{BACKEND_NAME}-compile", action=DriverName)

    # 2) The selected target name, propagated by onecc automatically when
    #    [backend] target=... is present in the ini.
    parser.add_argument("--target", action=TargetOption)

    # 3) Regular options supported by your backend driver.
    #    - Optional form: starts with hyphen(s)
    #    - Positional form: no leading hyphen
    parser.add_argument("--verbose", action=NormalOption, dtype=bool)

    # Many ONE tools use *_path; you can support both "output" and "output_path".
    parser.add_argument("--output", "--output_path", action=NormalOption)

    # Input is typically positional. Support both "input" and "input_path" keys.
    parser.add_argument("input", "input_path", action=NormalOption)

    # Add more backend-specific knobs as needed. Examples:
    # parser.add_argument("--opt-level", action=NormalOption, dtype=int)
    # parser.add_argument("--emit-debug-info", action=NormalOption, dtype=bool)
    # parser.add_argument("--arch", action=NormalOption)  # if your backend distinguishes sub-arch

    return parser
