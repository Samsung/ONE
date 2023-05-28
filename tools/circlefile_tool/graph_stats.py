#!/usr/bin/python

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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


class GraphStats():
    def __init__(self):
        from collections import Counter
        from collections import defaultdict
        self.op_counts = Counter()
        self.filled_memory = 0
        self.total_memory = 0

    def accumulate_op_count(self, op_str, count):
        self.op_counts[op_str] += count

    def accumulate_filled_memory(self, size):
        self.filled_memory += size

    def accumulate_total_memory(self, size):
        self.total_memory += size

    def __iadd__(self, other):
        self.op_counts += other.op_counts
        self.filled_memory += other.filled_memory
        self.total_memory += other.total_memory
        return self


def PrintGraphStats(stats, verbose):
    print("Number of all operator types: {0}".format(len(stats.op_counts)))

    # Print op type stats
    for op_name in sorted(stats.op_counts.keys()):
        occur = stats.op_counts[op_name]
        optype_info_str = "\t{:38}: {:4}".format(op_name, occur)

        print(optype_info_str)

    summary_str = "{0:46}: {1:4}".format("Number of all operators",
                                         sum(stats.op_counts.values()))
    print(summary_str)
    print('')

    # Print memory stats
    from tensor_printer import ConvertBytesToHuman
    print("Expected TOTAL  memory: {0}".format(ConvertBytesToHuman(stats.total_memory)))
    print("Expected FILLED memory: {0}".format(ConvertBytesToHuman(stats.filled_memory)))
    print('')


def CalcGraphStats(op_parser):
    stats = GraphStats()

    for type_str, oper_list in op_parser.operators_per_type.items():
        # number of occurrence of this operator type
        occur = len(oper_list)
        stats.accumulate_op_count(type_str, occur)

        # this operator type can be computed?
        can_compute = oper_list[0].operation.can_compute

    total_memory = 0
    filled_memory = 0  # only memory for constant
    for tensor in op_parser.GetAllTensors():
        if tensor.tf_buffer.DataLength() != 0:
            filled_memory += tensor.memory_size
        total_memory += tensor.memory_size
    stats.accumulate_filled_memory(filled_memory)
    stats.accumulate_total_memory(total_memory)

    return stats
