# Copyright 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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

from TestUtil import *

from circle import ActivationFunctionType
from circle import BuiltinOptions
from circle import BuiltinOperator
from circle import AddOptions
from circle import Conv2DOptions
from circle import DepthwiseConv2DOptions
from circle import FullyConnectedOptions
from circle import TransposeConvOptions
from circle import InstanceNormOptions
""""Test for a model with a single operator"""


class SingleOperatorTest(object):
    def DefaultOpPost(self, name, opcode, inputs, output):
        raise SystemExit('NYI operator: ' + str(opcode))

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

    def testAdd(self, fused_act):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(), BuiltinOperator.BuiltinOperator.ADD)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.AddOptions)

        self._opt = self._op.BuiltinOptions()
        opt = AddOptions.AddOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        checkActivation(fused_act, opt.FusedActivationFunction())

    def AddPre(self, name, x, y, fused_act):
        self.testAdd(fused_act)

    def AddPost(self, name, x, y, output, fused_act):
        self.testAdd(fused_act)

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

    def testDepthwiseConv2D(self, padding, stride, depth_multiplier, dilation, fused_act):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(),
                    BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions)

        self._opt = self._op.BuiltinOptions()
        opt = DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        checkPadding(padding, opt.Padding())
        assertTrue(opt.StrideW() == stride['w'], "Stride_w mismatches")
        assertTrue(opt.StrideH() == stride['h'], "Stride_h mismatches")
        assertTrue(opt.DepthMultiplier() == depth_multiplier,
                   "Depth multiplier mismatches")
        assertTrue(opt.DilationWFactor() == dilation['w'], "Dilation_w mismatches")
        assertTrue(opt.DilationHFactor() == dilation['h'], "Dilation_w mismatches")
        checkActivation(fused_act, opt.FusedActivationFunction())

    def DepthwiseConv2DPre(self, name, input, filter, bias, padding, stride,
                           depth_multiplier, dilation, fused_act):
        self.testDepthwiseConv2D(padding, stride, depth_multiplier, dilation, fused_act)

    def DepthwiseConv2DPost(self, name, input, filter, bias, padding, stride,
                            depth_multiplier, dilation, output, fused_act):
        self.testDepthwiseConv2D(padding, stride, depth_multiplier, dilation, fused_act)

    def testFullyConnected(self, fused_act):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(), BuiltinOperator.BuiltinOperator.FULLY_CONNECTED)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.FullyConnectedOptions)

        self._opt = self._op.BuiltinOptions()
        opt = FullyConnectedOptions.FullyConnectedOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        checkActivation(fused_act, opt.FusedActivationFunction())

    def FullyConnectedPre(self, name, input, weights, bias, fused_act):
        self.testFullyConnected(fused_act)

    def FullyConnectedPost(self, name, input, weights, bias, output, fused_act):
        self.testFullyConnected(fused_act)

    def testTransposeConv(self, padding, stride):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(), BuiltinOperator.BuiltinOperator.TRANSPOSE_CONV)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.TransposeConvOptions)

        self._opt = self._op.BuiltinOptions()
        opt = TransposeConvOptions.TransposeConvOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        checkPadding(padding, opt.Padding())
        assertTrue(opt.StrideW() == stride['w'], "Stride_w mismatches")
        assertTrue(opt.StrideH() == stride['h'], "Stride_h mismatches")

    def TransposeConvPre(self, name, input, filter, output_shape, bias, padding, stride):
        self.testTransposeConv(padding, stride)

    def TransposeConvPost(self, name, input, filter, output_shape, bias, padding, stride,
                          output):
        self.testTransposeConv(padding, stride)

    def testInstanceNorm(self, epsilon):
        # Check opcode
        opcode = self._model.OperatorCodes(self._op.OpcodeIndex())
        checkOpcode(opcode.BuiltinCode(), BuiltinOperator.BuiltinOperator.INSTANCE_NORM)

        # Check option
        checkBuiltinOptionType(self._op.BuiltinOptionsType(),
                               BuiltinOptions.BuiltinOptions.InstanceNormOptions)

        self._opt = self._op.BuiltinOptions()
        opt = InstanceNormOptions.InstanceNormOptions()
        opt.Init(self._opt.Bytes, self._opt.Pos)
        assertTrue(opt.Epsilon() == epsilon, "epsilon mismatches")

    def InstanceNormPre(self, name, input, gamma, beta, epsilon):
        self.testInstanceNorm(epsilon)

    def InstanceNormPost(self, name, input, gamma, beta, epsilon, output):
        self.testInstanceNorm(epsilon)
