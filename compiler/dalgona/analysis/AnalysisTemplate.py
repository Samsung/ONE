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
"""
A template for anlaysis code.
This template shows how to access the information of each operator inside hooks.
Users can write their own hooks by modifying this file.

NOTE See "Arguments of Hooks" section in README to understand argument types (Tensor, Stride, ..)
NOTE See "tests/SingleOperatorTest.py" for more operators.
"""


class AnalysisTemplate(object):
    def StartAnalysis(self, args: str):
        """
        Called when the analysis starts
        args: string given by --analysis_args option
        """
        print("Analysis started.")
        print("args", args)

    def EndAnalysis(self):
        """
        Called when the analysis ends
        """
        print("Analysis ended.")

    def StartNetworkExecution(self, inputs: list):
        """
        Called when the execution of a network starts
        inputs: list of Tensor
        """
        print("Network execution started.")

    def EndNetworkExecution(self, outputs: list):
        """
        Called when the execution of a network ends
        outputs: list of Tensor
        """
        print("Network execution ended.")

    def DefaultOpPre(self, name: str, opcode: str, inputs: list):
        """
        Default hook called before an operator is executed
        name: output tensor name (string)
        opcode: opcode name (string)
        inputs: list of Tensor
        """
        print("name", name)
        print("opcode", opcode)
        print("inputs", inputs)

    def DefaultOpPost(self, name: str, opcode: str, inputs: list, output: dict):
        """
        Default hook called after an operator is executed
        name: output tensor name (string)
        opcode: opcode name (string)
        inputs: list of Tensor
        output: Tensor
        """
        print("name", name)
        print("opcode", opcode)
        print("inputs", inputs)
        print("output", output)

    def Conv2DPre(self, name: str, input: dict, filter: dict, bias: dict, padding: str,
                  stride: dict, dilation: dict, fused_act: str):
        """
        Called before Conv2D layer execution
        name: output tensor name (string)
        opcode: opcode name (string)
        input: Tensor
        filter: Tensor
        bias: Tensor
        padding: Padding (string)
        stride: Stride
        dilation: Dilation
        fused_act: Fused activation functions (string)
        """
        print("name", name)
        print("input", input)
        print("filter", filter)
        print("bias", bias)
        print("padding", padding)
        print("stride", stride)
        print("dilation", dilation)
        print("fused activation", fused_act)

    def Conv2DPost(self, name: str, input: dict, filter: dict, bias: dict, padding: str,
                   stride: dict, dilation: dict, output: dict, fused_act: str):
        """
        Called after Conv2D layer execution
        name: output tensor name (string)
        opcode: opcode name (string)
        input: Tensor
        filter: Tensor
        bias: Tensor
        padding: Padding (string)
        stride: Stride
        dilation: Dilation
        output: Tensor
        fused_act: Fused activation functions (string)
        """
        print("name", name)
        print("input", input)
        print("filter", filter)
        print("bias", bias)
        print("padding", padding)
        print("stride", stride)
        print("dilation", dilation)
        print("output shape", output['data'].shape)
        print("output type", output['data'].dtype)
        print("fused activation", fused_act)
