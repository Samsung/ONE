#!/usr/bin/env python3

# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""""Test for a model with a single operator"""

from TestUtil import *

from circle import Model
from circle import BuiltinOptions
from circle import BuiltinOperator
from circle import Conv2DOptions


class SingleOperatorTest(object):
    def StartAnalysis(self, args):
        """Called when the analysis starts"""
        with open(args, 'rb') as f:
            buffer = f.read()
            self._model = Model.Model.GetRootAsModel(buffer, 0)

        # Check model has one subgraph
        assertTrue(self._model.SubgraphsLength() == 1, "Model has more than one subgraph")
        graph = self._model.Subgraphs(0)

        # Check model has one operator
        assertTrue(graph.OperatorsLength() == 1, "Model has more than one operator")
        self._op = graph.Operators(0)

    def DefaultOpPost(self, name, opcode, inputs, output):
        raise SystemExit('NYI operator: ' + str(opcode))

    def testConv2D(self, padding, stride, dilation, fused_act):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(), BuiltinOperator.BuiltinOperator.CONV_2D)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.Conv2DOptions)

        self._opt = self._op.BuiltinOptions()
        opt = Conv2DOptions.Conv2DOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        checkPadding(padding, opt.Padding())
        assertTrue(opt.StrideW() == stride['w'], "Stride_w mismatches")
        assertTrue(opt.StrideH() == stride['h'], "Stride_h mismatches")
        assertTrue(opt.DilationWFactor() == dilation['w'], "Dilation_w mismatches")
        assertTrue(opt.DilationHFactor() == dilation['h'], "Dilation_w mismatches")
        checkActivation(fused_act, opt.FusedActivationFunction())

    def Conv2DPre(self, name, input, filter, bias, padding, stride, dilation, fused_act):
        self.testConv2D(padding, stride, dilation, fused_act)

    def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output,
                   fused_act):
        self.testConv2D(padding, stride, dilation, fused_act)
