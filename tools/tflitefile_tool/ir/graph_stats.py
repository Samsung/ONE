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


def CalcGraphStats(subg):
    stats = GraphStats()

    for type_str, oper_list in subg.optypes_map.items():
        # number of occurrence of this operator type
        occur = len(oper_list)
        stats.accumulate_op_count(type_str, occur)

    total_memory = 0
    filled_memory = 0  # only memory for constant
    for index, tensor in subg.tensors_map.items():
        if tensor.buffer is not None:
            filled_memory += tensor.memory_size
        total_memory += tensor.memory_size
    stats.accumulate_filled_memory(filled_memory)
    stats.accumulate_total_memory(total_memory)

    return stats
