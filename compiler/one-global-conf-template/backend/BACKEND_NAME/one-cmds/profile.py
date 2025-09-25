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

# Command schema for one-profile when BACKEND={BACKEND_NAME}.
#
# Install path (required by ONE):
#   /usr/share/one/backends/command/{BACKEND_NAME}/profile.py
#
# Example onecc snippet:
#   [one-profile]
#   backend={BACKEND_NAME}
#   input_path=model.opt.circle
#   report_path=report.json
#   warmup=5
#   repeat=50

from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()

    # Backend driver for profiling
    parser.add_argument("{BACKEND_NAME}-profile", action=DriverName)

    # Target propagation (same as in codegen)
    parser.add_argument("--target", action=TargetOption)

    # Common profiling knobs
    parser.add_argument("input", "input_path", action=NormalOption)
    parser.add_argument("--report", "--report_path", action=NormalOption)
    parser.add_argument("--warmup", action=NormalOption, dtype=int)
    parser.add_argument("--repeat", action=NormalOption, dtype=int)

    # Optional: hardware execution switches
    # parser.add_argument("--device", action=NormalOption, dtype=int)
    # parser.add_argument("--dump-timeline", action=NormalOption, dtype=bool)

    return parser
