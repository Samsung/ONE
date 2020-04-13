#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
from operation import Operation


class PerfPredictor(object):
    def __init__(self, add_cycle=1, mul_cycle=1, nonlinear_cycle=1):
        self.add_cycle = add_cycle
        self.mul_cycle = mul_cycle
        self.nonlinear_cycle = nonlinear_cycle

    def PredictCycles(self, operation):
        return (operation.add_instr_num * self.add_cycle +
                operation.mul_instr_num * self.mul_cycle +
                operation.nonlinear_instr_num * self.nonlinear_cycle)
