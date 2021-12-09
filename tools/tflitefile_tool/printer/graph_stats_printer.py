#!/usr/bin/python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

from .tensor_printer import ConvertBytesToHuman


def GetStringGraphStats(stats):
    results = []

    results.append("Number of all operator types: {}".format(len(stats.op_counts)))

    # op type stats
    for op_name in sorted(stats.op_counts.keys()):
        occur = stats.op_counts[op_name]
        optype_info_str = "\t{:38}: {:4}".format(op_name, occur)
        results.append(optype_info_str)

    summary_str = "{0:46}: {1:4}".format("Number of all operators",
                                         sum(stats.op_counts.values()))
    results.append(summary_str)
    results.append('\n')

    # memory stats
    results.append("Expected TOTAL  memory: {}".format(
        ConvertBytesToHuman(stats.total_memory)))
    results.append("Expected FILLED memory: {}".format(
        ConvertBytesToHuman(stats.filled_memory)))
    results.append('\n')

    return "\n".join(results)


# TODO: Extract to a single Printer class like Printer.print(stats)
def PrintGraphStats(stats, verbose):
    print(GetStringGraphStats(stats))
