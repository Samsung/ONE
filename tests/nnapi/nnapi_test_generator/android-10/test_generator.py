#!/usr/bin/python3

# Copyright 2017, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NN model compiler

Contain classes definition and utilify functions for compiling models and
examples into NDK-based CTS and VTS unit tests.

Used by cts_generator.py, vts_generator.py, and slicing.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import copy
from functools import reduce
import itertools
import math
import os
import re
import struct
import sys
import contextlib
import pprint
import numpy as np

def GetJointStr(l, sep=", ", method=str):
    return sep.join([method(i) for i in l])

# Print in C float literal format
def PrettyPrintAsFloat(x):
    s = str(float(x))
    if s.find(".") >= 0 or s.find("e") >= 0:
        return s + "f"
    else:
        return s + ".0f"

# Transform from original type to float32
def Dequantize(v, ty):
    v -= ty.zeroPoint
    if ty.scale != 0:
        v *= ty.scale
    if isinstance(ty.extraParams, SymmPerChannelQuantParams):
        v *= ty.extraParams.GetScalesBroadcastArray(ty.dimensions)
    return v

# Transform float32 to target data type
def Quantize(v, ty):
    if ty.scale != 0:
        v /= ty.scale
    if isinstance(ty.extraParams, SymmPerChannelQuantParams):
        v = v / ty.extraParams.GetScalesBroadcastArray(ty.dimensions)
    v += ty.zeroPoint
    if not ty.IsFloat():
        v = np.round(v)
        v = int(v) if np.isscalar(v) else v.astype(int)
    if ty.type == "TENSOR_QUANT8_ASYMM":
        v = np.minimum(np.maximum(v, 0), 255)
    elif ty.type == "TENSOR_QUANT16_ASYMM":
        v = np.minimum(np.maximum(v, 0), 65535)
    elif ty.type == "TENSOR_QUANT8_SYMM_PER_CHANNEL":
        v = np.minimum(np.maximum(v, -127), 127)
    elif ty.type == "UINT32":
        v = np.maximum(v, 0)
    return v

@contextlib.contextmanager
def SmartOpen(filename=None, mode="w"):
    if filename and filename != '-':
        fh = open(filename, mode)
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

# Tracking objects inside a model with a unique name
class NamedObject:
    existingNames = set()

    def __init__(self, *args, sep="_", showZero=False, startsFrom=0, skipRenaming=False):
        name = GetJointStr([i for i in args if i is not None and i != ""], sep=sep)
        if skipRenaming:
            self.name = name
            return
        # make the name unique by renaming with a suffix number
        uniqueName = name if showZero is False else name + sep + str(startsFrom)
        while uniqueName in self.__class__.existingNames:
            startsFrom += 1
            uniqueName = name + sep + str(startsFrom)
        self.__class__.existingNames.add(uniqueName)
        self.name = uniqueName

    def __str__(self):
        return self.name
    __repr__ = __str__

    # Since names are unique, objects with the same name are considered equal
    def __eq__(self, other):
        return isinstance(other, NamedObject) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

# Types, operands should all have a unique name since they share the same namespace
class NamedVariable(NamedObject):
    existingNames = set()
    def __init__(self, *args, sep="_", showZero=False, startsFrom=0, skipRenaming=False):
        NamedObject.__init__(self, *args, sep=sep, showZero=showZero,
            startsFrom=startsFrom, skipRenaming=skipRenaming)

# Global variables in the spec namespace such as CreateModel, is_ignored, and examples
class GlobalVariable(NamedVariable):
    def __init__(self, *args, skipRenaming=False):
        NamedObject.__init__(self, *args, startsFrom=1, skipRenaming=skipRenaming)

# Each test should have a unique name, but will not conflict with variables
class NamedTest(NamedObject):
    existingNames = set()
    def __init__(self, *args, startsFrom=0, skipRenaming=False):
        NamedObject.__init__(self, *args, startsFrom=1, skipRenaming=skipRenaming)

class Type(NamedVariable):
    typesMap = dict()
    typeLookup = {
        "INT32": "int32_t",
        "UINT32": "uint32_t",
        "FLOAT32": "float",
        "FLOAT16": "_Float16",
        "TENSOR_INT32": "int32_t",
        "TENSOR_FLOAT16": "_Float16",
        "TENSOR_FLOAT32": "float",
        "TENSOR_QUANT8_ASYMM": "uint8_t",
        "TENSOR_QUANT8_SYMM": "int8_t",
        "BOOL": "bool8",
        "TENSOR_QUANT16_ASYMM": "uint16_t",
        "TENSOR_QUANT16_SYMM": "int16_t",
        "TENSOR_BOOL8": "bool8",
        "TENSOR_QUANT8_SYMM_PER_CHANNEL": "int8_t",
#     "OEM_SCALAR": this is service-defined.
        "TENSOR_OEM_BYTE": "uint8_t",
    }

    # types are named as "type0", "type1", ...
    def __init__(self, vt, dimensions, scale, zeroPoint, name="type", skipRenaming=False,
                 extraParams=None):
        NamedVariable.__init__(self, name, sep="", showZero=True, skipRenaming=skipRenaming)
        self.type = vt
        self.dimensions = dimensions
        self.scale = float(scale)
        self.zeroPoint = int(zeroPoint)
        self.extraParams = extraParams

    # Factory for Type object, only create a new Type if requested type does
    # not have a match with all existing types
    @staticmethod
    def GetType(vt, dimensions, scale=0, zeroPoint=0, extraParams=None):
        key = ",".join([vt, str(dimensions), str(scale), str(zeroPoint), str(extraParams)])
        if key not in Type.typesMap:
            Type.typesMap[key] = Type(vt, dimensions, scale, zeroPoint, extraParams=extraParams)
        return Type.typesMap[key]

    @staticmethod
    def GetAllTypes():
        # sort to ensure a stable order when dumping the code
        return sorted(Type.typesMap.values())

    # For backward-compatibility
    @staticmethod
    def GetTypeFromString(vt, shape, extraParams=None):
        dimensions, scale, zeroPoint = Type.GetParsedShape(shape)
        scale = float(scale)
        zeroPoint = int(zeroPoint)
        return Type.GetType(vt, dimensions, scale, zeroPoint, extraParams)

    # For backward-compatibility
    @staticmethod
    def GetParsedShape(shape):
        # Parse shape
        if (shape != "" and shape != "{}"):
            left, sep, right = shape.partition('{')
            real_shape, sep, right = right.partition('}')
            shape = [int(x) for x in real_shape.split(",")]
            # left now looks like "0.0f, 127.5f, "
            scale, sep, zero_point = right.rpartition(',')
            if scale == "":
                if zero_point == "":
                    return shape, "0", "0"
                return shape, zero_point, "0"
            left, sep, scale = scale.partition(',')
            return shape, scale.replace("f", ""), zero_point
        else:
            return [], "0", "0"

    def GetNumberOfElements(self):
        return reduce(lambda x,y: x*y, self.dimensions, 1)

    def GetCppTypeString(self):
        return Type.typeLookup[self.type]

    def IsFloat(self):
        return self.GetCppTypeString() in ["float", "_Float16"]

    def IsBool(self):
        return self.GetCppTypeString() == "bool8"

    def GetElementByteSize(self):
        cppTypeString = self.GetCppTypeString()
        if cppTypeString in ["uint8_t", "int8_t", "bool8"]:
            return 1
        elif cppTypeString in ["int16_t", "uint16_t", "_Float16"]:
            return 2
        else:
            return 4

    def GetByteSize(self):
        return self.GetElementByteSize() * self.GetNumberOfElements()

    def GetDimensionsString(self):
        return "{" + GetJointStr(self.dimensions) + "}"

    def GetSignatureTuple(self):
        return (self.type, self.dimensions, self.scale, self.zeroPoint)

    # For backward-compatibility with slicing.py
    def GetRawShape(self):
        if self.scale == 0 and self.zeroPoint == 0:
            return self.GetDimensionsString()
        else:
            return GetJointStr([self.GetDimensionsString(), self.scale, self.zeroPoint])

    def ToUnspecifiedDim(self):
        return Type.GetType(self.type, [0] * len(self.dimensions), self.scale, self.zeroPoint)

# To track implicitly convertible parameter types
class ImplicitParameter():
    @staticmethod
    def ImplicitConvertion(value):
        if isinstance(value, Operand):
            return value
        for implicitType in ImplicitParameter.__subclasses__():
            if implicitType.IsCompatible(value):
                return implicitType("param", value)
        assert False, "%s not supported for implicit parameter"%value


# ExtraParams with per-channel quantization.
class SymmPerChannelQuantParams():
  def __init__(self, channelDim, scales, hide = False):
    self.channelDim = channelDim
    self.scales = scales
    self.hide = hide

  def GetScalesBroadcastArray(self, dimensions):
    bshape = [1] * len(dimensions)
    bshape[self.channelDim] = len(self.scales)
    return np.array(self.scales).reshape(bshape)

  def GetConstructor(self):
    return "SymmPerChannelQuantParams({%s},%d)" % (
        ", ".join(str(x) + "f" for x in self.scales), self.channelDim)

  def GetVtsSetter(self):
    return "channelQuant"

  def GetVtsConstructor(self):
    return "SymmPerChannelQuantParams{.scales={%s}, .channelDim=%d}" % (
        ", ".join(str(x) + "f" for x in self.scales), self.channelDim)


# An operand that can be fed into operations. Also, an operand is always
# declared before operations.
class Operand(NamedVariable):

    def __init__(self, name, opType, value, backward=None, skipRenaming=False, extraParams=None):
        NamedVariable.__init__(self, name, sep="", skipRenaming=skipRenaming)
        if type(opType) is str:
            self.type = Type.GetTypeFromString(opType, value, extraParams)
            value = backward
        else:
            self.type = Type.GetType(*opType, extraParams=extraParams)
        self.SetValue(value)
        self.dimensions = self.type.dimensions
        self.lifetime = "TEMPORARY_VARIABLE"
        self.ins = []
        self.outs = []

    def SetValue(self, value):
        self.value = value if type(value) is list or type(value) is tuple else [value]
        return self

    def SetValueFromNumpy(self, value):
        self.value = value.flatten().tolist()
        return self

    def GetValueAsNumpy(self):
        return np.array(self.value).reshape(self.type.dimensions)

    # Print value as cpp-style list initialization
    def GetListInitialization(self):
        assert self.value is not None, \
            "Trying to print operand %s with None value"%(str(self))
        if self.type.IsFloat():
            return "{%s}"%(GetJointStr(self.value, method=PrettyPrintAsFloat))
        elif self.type.IsBool():
            return "{%s}"%(GetJointStr(self.value, method=lambda v: "true" if v else "false"))
        else:
            return "{%s}"%(GetJointStr(self.value, method=lambda x: str(int(x))))

    def ToUnspecifiedDim(self):
        self.dimensions = self.type.dimensions
        self.type = self.type.ToUnspecifiedDim()

# Base class of user-defined input/output operand
class InOut(Operand):

    def __init__(self, name, opType, backward=None, skipRenaming=False, extraParams=None):
        Operand.__init__(self, name, opType, backward, None, skipRenaming=skipRenaming, extraParams=extraParams)
        self.lifetime = "MODEL_INPUT"
        self.index = 0

    def Feed(self, value):
        self.SetValue(value[self] if type(value) is dict else value)
        return self

    def GetListInitialization(self):
        return "{%d, %s}"%(self.index, super().GetListInitialization())

# A user-declared input operand
class Input(InOut):
    def __init__(self, name, opType, backward=None, skipRenaming=False, extraParams=None):
        InOut.__init__(self, name, opType, backward, skipRenaming=skipRenaming, extraParams=extraParams)
        self.lifetime = "MODEL_INPUT"

# A user-declared output operand
class Output(InOut):
    def __init__(self, name, opType, backward=None, skipRenaming=False):
        InOut.__init__(self, name, opType, backward, skipRenaming=skipRenaming)
        self.lifetime = "MODEL_OUTPUT"

# An output that we don't want to compare the results
class IgnoredOutput(Output):
    def __init__(self, name, opType, backward=None, skipRenaming=False):
        Output.__init__(self, name, opType, backward, skipRenaming=skipRenaming)
        self.lifetime = "MODEL_OUTPUT"
    def Feed(self, value):
        numElements = reduce(lambda x,y: x*y, self.dimensions, 1)
        self.value = [0 for x in range(numElements)]
        return self

# An explicitly declared parameter
class Parameter(Operand):
    def __init__(self, name, opType, value, backward=None, skipRenaming=False, extraParams=None):
        Operand.__init__(self, name, opType, value, backward, skipRenaming=skipRenaming,
                         extraParams=extraParams)
        self.initializer = NamedVariable(str(self) + "_init")
        self.lifetime = "CONSTANT_REFERENCE" if Configuration.useSHM() else "CONSTANT_COPY"

# A shortcut for parameters of INT32
class Int32Scalar(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("INT32", []), int(value))
    @staticmethod
    def IsCompatible(value):
        return type(value) is int

# A shortcut for parameters of FLOAT16
class Float16Scalar(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("FLOAT16", []), float(value))
    @staticmethod
    def IsCompatible(value):
        return False

# A shortcut for parameters of FLOAT32
class Float32Scalar(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("FLOAT32", []), float(value))
    @staticmethod
    def IsCompatible(value):
        return type(value) is float

# A shortcut for parameters of BOOL
class BoolScalar(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("BOOL", []), bool(value))
    @staticmethod
    def IsCompatible(value):
        return type(value) is bool

# A shortcut for parameter of 1-D TENSOR_INT32
class Int32Vector(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("TENSOR_INT32", [len(value)]), [int(v) for v in value])
    @staticmethod
    def IsCompatible(value):
        if type(value) is not list and type(value) is not tuple:
            return False
        return all(type(i) is int for i in value)

# A shortcut for parameter of 1-D TENSOR_FLOAT32
class Float32Vector(Parameter, ImplicitParameter):
    def __init__(self, name, value):
        Parameter.__init__(self, name, ("TENSOR_FLOAT32", [len(value)]), [float(v) for v in value])
    @staticmethod
    def IsCompatible(value):
        if type(value) is not list and type(value) is not tuple:
            return False
        return all(type(i) is float for i in value)

# An explicitly declared intermediate result
class Internal(Operand):
    def __init__(self, name, opType, backward=None, skipRenaming=False):
        Operand.__init__(self, name, opType, backward, None, skipRenaming=skipRenaming)
        self.lifetime = "TEMPORARY_VARIABLE"

# An operation in a model, does not need a name
class Operation:

    def __init__(self, optype, ins, outs):
        self.optype = optype
        self.SetInputs(ins)
        self.SetOutputs(outs)

    # for the ease of debugging
    def __str__(self):
        insString = GetJointStr(self.ins)
        outsString = GetJointStr(self.outs)
        return "Operation %s: [%s] -> [%s]"%(self.optype, insString, outsString)
    __repr__ = __str__

    def SetInputs(self, ins):
        self.ins = [ImplicitParameter.ImplicitConvertion(i) for i in ins]
        return self

    def SetOutputs(self, outs):
        self.outs = list(outs)
        return self

    # For backward-compatibility with slicing.py
    # Get Python-ish dump for the op
    def PyDefinition(self):
        py_op_string = """Operation("{optype}", {inputs}).To({outputs})"""
        inputs = [str(x) for x in self.ins]
        inputs = ", ".join(inputs)
        assert len(self.outs) <= 1
        outputs = str(self.outs[0])
        ops = {"optype": self.optype, "inputs": inputs, "outputs": outputs}
        return py_op_string.format(**ops)

# Main interface
class Model:
    models = list()

    def __init__(self, name=None):
        self.name = name
        self.operations = []
        self.operands = []
        self.isRelaxed = False
        self.compiled = False
        self.dumped = False
        self.hasDynamicOutputShape = False
        self.version = FileNames.version
        Model.models.append(self)

    def WithSuffix(self, *args):
        self.createFunctionName = GlobalVariable("CreateModel", self.name, *args)
        self.createTestFunctionName = GlobalVariable("createTestModel", self.name, *args)
        self.isIgnoredFunctionName = GlobalVariable("is_ignored", self.name, *args)
        return self

    def AddOperation(self, operation):
        self.operations.append(operation)
        for i in operation.ins:
            if i not in self.operands:
                self.operands.append(i)
        for o in operation.outs:
            if o not in self.operands:
                self.operands.append(o)
        return self

    def Operation(self, op_name, *args):
        return self.AddOperation(Operation(op_name, args, []))

    def To(self, *args):
        assert len(self.operations) > 0
        if type(args[0]) is tuple or type(args[0]) is list:
            outs = args[0]
        else:
            outs = args
        self.operations[-1].SetOutputs(outs)
        for o in outs:
            if o not in self.operands:
                self.operands.append(o)
        return self

    def RelaxedExecution(self, isRelaxed):
        self.isRelaxed = isRelaxed
        return self

    def TestDynamicOutputShape(self, hasDynamicOutputShape):
        self.hasDynamicOutputShape = hasDynamicOutputShape
        return self

    # Sets the version of the model in compliance tests. Set to None to disable the test.
    def IntroducedIn(self, ver):
        self.version = ver
        return self

    def GetTypes(self):
        return sorted(list(set(op.type for op in self.operands)))

    def GetInputs(self):
        return [i for i in self.operands if isinstance(i, Input)]

    def GetOutputs(self):
        return [o for o in self.operands if isinstance(o, Output)]

    def GetInputsIndex(self):
        return [i for i,op in enumerate(self.operands) if isinstance(op, Input)]

    def GetOutputsIndex(self):
        return [o for o,op in enumerate(self.operands) if isinstance(op, Output)]

    def GetIndexOfOperands(self, operands):
        return [self.operands.index(i) for i in operands]

    def GetIgnoredOutputs(self):
        return [o for o in self.operands if isinstance(o, IgnoredOutput)]

    def GetParameters(self):
        return [p for p in self.operands if isinstance(p, Parameter)]

    def GetEquivalentOperands(self, targets):
        return [self.operands[self.operands.index(t)] for t in targets]

    def UpdateEquivalentOperands(self, targets):
        for t in targets:
            self.operands[self.operands.index(t)] = t
        return self

    def SetInputAndOutputIndex(self):
        for ind, i in enumerate(self.GetInputs()):
            i.index = ind
        for ind, o in enumerate(self.GetOutputs()):
            o.index = ind
        return self

    def SetOperandInsAndOuts(self):
        for op in self.operands:
            op.ins = list()
            op.outs = list()
        for op in self.operations:
            op.ins = self.GetEquivalentOperands(op.ins)
            op.outs = self.GetEquivalentOperands(op.outs)
            for i in op.ins:
                i.outs.append(op)
            for o in op.outs:
                o.ins.append(op)
        return self

    def TopologicalSortHelper(self, op, deps, visited):
        if op in visited:
            assert op not in deps, "Cycle detected in the graph"
        else:
            visited.add(op)
            for i in deps[op]:
                self.TopologicalSortHelper(i, deps, visited)
            self.operations.append(op)
            deps.pop(op)

    # Topological sort of the operations, and detect if there is a cycle is the graph
    def TopologicalSort(self):
        deps = {op: list() for op in self.operations}
        [deps[o].append(i) for op in self.operands for o in op.outs for i in op.ins]
        operations = self.operations.copy()
        self.operations = []
        visited = set()
        for op in operations:
            self.TopologicalSortHelper(op, deps, visited)

    def SetOutputUnspecified(self):
        for op in self.operands:
            op.dimensions = op.type.dimensions
        if self.hasDynamicOutputShape:
            for op in self.GetOutputs():
                op.ToUnspecifiedDim()
        return self

    def Compile(self):
        if self.compiled:
            return self
        self.SetInputAndOutputIndex()
        self.SetOperandInsAndOuts()
        self.TopologicalSort()
        self.SetOutputUnspecified()
        # Do not check compliance for relaxed mode and dynamic output shape tests.
        if self.isRelaxed or self.hasDynamicOutputShape:
            self.IntroducedIn(None)
        self.compiled = True
        return self

# To track implicitly convertible variation types
class ImplicitVariation:
    @staticmethod
    def ImplicitConvertion(value):
        if isinstance(value, ModelVariation):
            return value
        for implicitType in ImplicitVariation.__subclasses__():
            value = value if type(value) is tuple or type(value) is list else [value]
            if implicitType.IsCompatible(value[0]):
                var = implicitType(value[0])
                if len(value) > 1:
                    var.Identify(*value[1:])
                return var
        assert False, "%s not supported for implicit variation"%value[0]

# The base class for model variations
class ModelVariation:

    def __init__(self, name=None):
        self.targetOperands = {}
        self.name = name

    def ApplyToHelper(self, model, args, feedDicts, transform):
        opVarList = []
        for op in model.GetEquivalentOperands(sorted(args.keys())):
            opVar = op
            feedDictsVar = []
            if isinstance(op, Input) or isinstance(op, Output):
                for feedDict in feedDicts:
                    op_tmp = copy.deepcopy(op)
                    if op_tmp in feedDict[0]:
                        opVar = transform(op_tmp.Feed(feedDict[0]), args[op_tmp])
                    elif op_tmp in feedDict[1]:
                        opVar = transform(op_tmp.Feed(feedDict[1]), args[op_tmp])
                    else:
                        assert False
                    feedDictsVar.append(opVar.value)
                assert type(op) == type(opVar), "Can not handle %s -> %s"%(type(op), type(opVar))
            else:
                opVar = transform(op, args[op])
                # handle Parameter -> Input
                if isinstance(opVar, Input) or isinstance(opVar, Output):
                    feedDictsVar = [opVar.value] * len(feedDicts)
            if isinstance(opVar, Input) or isinstance(opVar, Output):
                for feedDict, feedDictVar in zip(feedDicts, feedDictsVar):
                    if opVar in feedDict[1]:
                        feedDict[1][opVar] = feedDictVar
                    else:
                        feedDict[0][opVar] = feedDictVar
            opVarList.append(opVar)
        return opVarList

    # Make a deepcopy of the model and feedDicts, and apply the change
    def ApplyTo(self, modelOrigin, feedDictsOrigin):
        model, feedDicts = copy.deepcopy((modelOrigin, feedDictsOrigin))
        model.compiled = False
        model.dumped = False

        if not self.targetOperands:
            self.AutoIdentify(model)

        # get transformed operands and update feedDicts
        operandsVar = self.ApplyToHelper(
            model, self.targetOperands, feedDicts, self.TransformOperand)

        model = self.TransformModel(model)
        model.UpdateEquivalentOperands(operandsVar)
        return model, feedDicts

    def IdentifyOperands(self, args=None):
        if args is None:
            return self
        self.targetOperands = args if type(args) is dict else {i: None for i in args}
        return self

    def Identify(self, operandArgs=None, paramArgs=None):
        self.IdentifyOperands(operandArgs)
        return self

    # Set variation to its default name
    def SetToDefaultName(self):
        self.name = ""
        return self

    # Automatically select the target operand list
    def AutoIdentify(self, model):
        return self

    # Transform operands that are marked by IdentifyOperands()
    def TransformOperand(self, op, arg=None):
        return op

    # Transform the model
    def TransformModel(self, model):
        return model

# Default variation that does nothing
class DefaultVariation(ModelVariation):

    def __init__(self, name=None):
        ModelVariation.__init__(self, name=name)

# Convert operand data type
class DataTypeConverter(ModelVariation, ImplicitVariation):

    def __init__(self, targetType=None, name=None):
        ModelVariation.__init__(self, name=name)
        if targetType is not None:
            assert DataTypeConverter.IsCompatible(targetType)
        self.targetType = targetType

    @staticmethod
    def IsCompatible(value):
        return value.lower() in ["float16", "int32"]

    def SetToDefaultName(self):
        if self.targetType is not None:
            self.name = self.targetType.lower()
            return self
        # get all target types
        targetTypes = list(zip(*self.targetOperands.values()))[0]
        if "TENSOR_QUANT8_SYMM_PER_CHANNEL" in targetTypes:
            self.name = "channelQuant8"
        elif "TENSOR_QUANT8_ASYMM" in targetTypes:
            self.name = "quant8"
        elif "TENSOR_INT32" in targetTypes:
            self.name = "int32"
        elif "TENSOR_FLOAT16" in targetTypes:
            self.name = "float16"
        else:
            self.name = "float32"
        return self

    def AutoIdentify(self, model):
        if self.targetType is not None:
            # By default, select all the float32 tensors/scalars
            targets = {op: ["TENSOR_" + self.targetType.upper()] \
                    for op in model.operands if op.type.type == "TENSOR_FLOAT32"}
            targets.update({op: [self.targetType.upper()] \
                    for op in model.operands if op.type.type == "FLOAT32"})
            self.Identify(targets)
        return self

    def TransformOperand(self, op, arg=None):
        if len(arg) == 1:
            typeTuple = (arg[0], op.type.dimensions)
        else:
            typeTuple = (arg[0], op.type.dimensions, *arg[1:])
        # To handle Internal operands
        if op.value is None or op.type.GetNumberOfElements() == 0:
            op.type = Type.GetType(*typeTuple)
        else:
            v = Dequantize(op.GetValueAsNumpy().astype(np.float32), op.type)
            op.type = Type.GetType(*typeTuple)
            v = Quantize(v, op.type)
            op.SetValueFromNumpy(v)
        return op

# Convert model to turn on/off relaxed computation
class RelaxedModeConverter(ModelVariation, ImplicitVariation):

    def __init__(self, isRelaxed=True, name=None):
        ModelVariation.__init__(self, name=name)
        if isinstance(isRelaxed, bool):
            self.isRelaxed = isRelaxed
        else:
            assert RelaxedModeConverter.IsCompatible(isRelaxed.lower())
            self.isRelaxed = True

    @staticmethod
    def IsCompatible(value):
        return value.lower() in ["relaxed"]

    def SetToDefaultName(self):
        self.name = "relaxed" if self.isRelaxed else "float"
        return self

    def TransformModel(self, model):
        model.RelaxedExecution(self.isRelaxed)
        return model

# Convert data layout between "NHWC" amd "NCHW"
class DataLayoutConverter(ModelVariation, ImplicitVariation):

    def __init__(self, targetLayout="nchw", name=None):
        ModelVariation.__init__(self, name=name)
        self.targetLayout = targetLayout.lower()
        assert DataLayoutConverter.IsCompatible(self.targetLayout)
        self.perm = (0, 3, 1, 2) if self.targetLayout == "nchw" else (0, 2, 3, 1)
        self.param = True if self.targetLayout == "nchw" else False

    @staticmethod
    def IsCompatible(value):
        return value.lower() in ["nhwc", "nchw"]

    def SetToDefaultName(self):
        self.name = self.targetLayout
        return self

    def TransformOperand(self, op, arg=None):
        if len(op.type.dimensions) == 4:
            # To handle Internal operands
            if op.value is not None and op.type.GetNumberOfElements() != 0:
                op.SetValueFromNumpy(op.GetValueAsNumpy().transpose(self.perm))
            newDim = [op.type.dimensions[i] for i in self.perm]
            op.type = Type.GetType(op.type.type, newDim, op.type.scale, op.type.zeroPoint)
        elif len(op.type.dimensions) == 1 and len(op.value) == 4:
            op.SetValueFromNumpy(op.GetValueAsNumpy()[list(self.perm)])
        elif op.type.type == "BOOL":
            op.SetValue(self.param)
        else:
            assert False, "%s not supported by DataLayoutConverter"%op
        return op

# Convert data by tansposing and removing axis
class AxisConverter(ModelVariation):

    def __init__(self, origin, target, dim, drop=[], name=None):
        ModelVariation.__init__(self, name=name)
        self.origin = origin
        self.target = target
        assert all(i >= -dim and i < dim for i in [self.origin, self.target])
        self.dim = dim
        self.perm = list(range(dim))
        self.perm.insert(target if target >= 0 else target + dim, self.perm.pop(origin))
        self.drop = [drop] if type(drop) is int else list(drop)
        assert all(i >= -dim and i < dim for i in self.drop)
        self.drop = [i if i >= 0 else i + dim for i in self.drop]
        assert target not in self.drop and target + dim not in self.drop

    def SetToDefaultName(self):
        axis = self.target if self.target >= 0 else self.target + self.dim
        axis -= sum(i < axis for i in self.drop)
        neg = "" if self.target >= 0 else "_neg"
        self.name = "dim%d_axis%d%s"%(self.dim - len(self.drop), axis, neg)
        return self

    def TransposeAxis(self, op):
        if op.type.type == "INT32":
            op.SetValue(self.target)
        elif len(op.type.dimensions) == self.dim:
            # To handle Internal operands
            if op.value is not None:
                op.SetValueFromNumpy(op.GetValueAsNumpy().transpose(self.perm))
            newDim = [op.type.dimensions[i] for i in self.perm]
            op.type = Type.GetType(op.type.type, newDim, op.type.scale, op.type.zeroPoint)
        else:
            assert False, "%s not supported by AxisConverter"%op
        return op

    def RemoveAxis(self, op):
        if op.type.type == "INT32":
            if op.value[0] >= 0:
                op.SetValue(op.value[0] - sum(i < op.value[0] for i in self.drop))
            else:
                op.SetValue(op.value[0] + sum(i > (op.value[0] + self.dim) for i in self.drop))
        elif len(op.type.dimensions) == self.dim:
            if op.value is not None:
                val = op.GetValueAsNumpy()
                for i in sorted(self.drop, reverse=True):
                    val = np.take(val, 0, axis=i)
                op.SetValueFromNumpy(val)
            newDim = [op.type.dimensions[i] for i in range(self.dim) if i not in self.drop]
            op.type = Type.GetType(op.type.type, newDim, op.type.scale, op.type.zeroPoint)
        else:
            assert False, "%s not supported by AxisConverter"%op
        return op

    def TransformOperand(self, op, arg=None):
        op = self.TransposeAxis(op)
        op = self.RemoveAxis(op)
        return op

# Convert a Parameter to Input
class ParameterAsInputConverter(ModelVariation, ImplicitVariation):

    def __init__(self, arg="as_input", prefix="weight", name=None):
        ModelVariation.__init__(self, name=name)
        assert ParameterAsInputConverter.IsCompatible(arg.lower())
        self.prefix = prefix

    @staticmethod
    def IsCompatible(value):
        return value.lower() in ["as_input"]

    def SetToDefaultName(self):
        self.name = self.prefix + "_as_input"
        return self

    def TransformOperand(self, op, arg=None):
        assert isinstance(op, Parameter), "%s cannot be converted to Input."%type(op)
        newop = Input(op.name, op.type.GetSignatureTuple(), skipRenaming=True, extraParams=op.type.extraParams)
        newop.SetValue(op.value)
        return newop

# Convert Output based on activation
class ActivationConverter(ModelVariation, ImplicitVariation):
    # (Enum, low, high)
    actMap = {
        "none": (0, None, None),
        "relu": (1, 0.0, None),
        "relu1": (2, -1.0, 1.0),
        "relu6": (3, 0.0, 6.0),
    }
    def __init__(self, act="relu", name=None):
        ModelVariation.__init__(self, name=name)
        self.act = act.lower()
        assert ActivationConverter.IsCompatible(self.act)
        self.enum = ActivationConverter.actMap[self.act][0]
        self.low = ActivationConverter.actMap[self.act][1]
        self.high = ActivationConverter.actMap[self.act][2]

    @staticmethod
    def IsCompatible(value):
        return value.lower() in ActivationConverter.actMap.keys()

    def SetToDefaultName(self):
        self.name = self.act
        return self

    def TransformOperand(self, op, arg=None):
        if op.type.type == "INT32": # activation enum
            return op.SetValue(self.enum)
        else:
            assert isinstance(op, Output)
            v = op.GetValueAsNumpy()
            if self.low is not None:
                low = Quantize(self.low, op.type)
                v = np.maximum(v, low)
            if self.high is not None:
                high = Quantize(self.high, op.type)
                v = np.minimum(v, high)
            return op.SetValueFromNumpy(v)

class DynamicOutputShapeConverter(ModelVariation):
    def __init__(self, name=None):
        ModelVariation.__init__(self, name=name)

    def SetToDefaultName(self):
        self.name = "dynamic_output_shape"
        return self

    def TransformModel(self, model):
        model.TestDynamicOutputShape(True)
        return model

# An example is always attached to a model, and could have multiple variations
class Example:
    examples = []
    versionOverrides = {}

    def __init__(self, *args, model=None, name=None):
        self.model = Model.models[-1] if model is None else model
        self.name = name
        self.expectedMultinomialDistributionTolerance = None
        self.feedDicts = []
        for feedDict in args:
            if type(feedDict) is tuple or type(feedDict) is list:
                self.feedDicts.append(feedDict)
            elif type(feedDict) is dict:
                self.feedDicts.append((
                    {i: feedDict[i] for i in self.model.GetInputs()},
                    {o: feedDict[o] for o in self.model.GetOutputs()}
                ))
            else:
                assert False
        # Fix for onert: disable dynamic shape test generation
        #if Configuration.test_dynamic_output_shape:
            #self.variations = [[DefaultVariation(), DynamicOutputShapeConverter()]]
        #else:
        self.variations = []
        Example.examples.append(self)

    @staticmethod
    def SetVersion(ver, *args):
        for name in args:
            Example.versionOverrides[name] = ver

    # Main entrance of test generator
    @staticmethod
    def DumpAllExamples(DumpModel=None, model_fd=None,
                        DumpExample=None, example_fd=None,
                        DumpTest=None, test_fd=None):
        Example.CombineAllExamples()
        for example in Example.examples:
            example.Dump(DumpModel, model_fd, DumpExample, example_fd, DumpTest, test_fd)

    # Combine examples with the same model, same name, and same set of variations
    @staticmethod
    def CombineAllExamples():
        modelMap = {}
        newExamples = []
        for example in Example.examples:
            key = (example.model, example.name, tuple(tuple(e) for e in example.variations))
            if key in modelMap:
                modelMap[key].Combine(example)
            else:
                modelMap[key] = example
                newExamples.append(example)
        Example.examples = newExamples

    def AddVariations(self, *args, includeDefault=True, defaultName=None):
        self.variations.append([DefaultVariation(defaultName)] if includeDefault else [])
        # NNFW Fix: remove float16 type variation test generation
        variations = []
        for i in args:
            variation = ImplicitVariation.ImplicitConvertion(i)
            print(i, file=sys.stderr)
            if not isinstance(i, ModelVariation) and type(i) is str:
              if i == "float16":
                continue
            else:
              variations.append(variation)
        self.variations[-1].extend(variations)
        #self.variations[-1].extend(ImplicitVariation.ImplicitConvertion(i) for i in args)
        return self

    def AddNchw(self, *args, includeDefault=True, defaultName="nhwc"):
        var = DataLayoutConverter("nchw").Identify(args)
        self.AddVariations(var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddRelaxed(self, isRelaxed=True, includeDefault=True, defaultName=None):
        var = RelaxedModeConverter(isRelaxed)
        self.AddVariations(var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddInput(self, *args, includeDefault=True, defaultName=None):
        var = ParameterAsInputConverter().Identify(args)
        self.AddVariations(var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddRelu(self, *args, includeDefault=True, defaultName=None):
        var = ActivationConverter("relu").Identify(args)
        self.AddVariations(var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddAllActivations(self, *args):
        var = [ActivationConverter(i).Identify(args)
            for i in sorted(ActivationConverter.actMap.keys())]
        self.AddVariations(*var, includeDefault=False)
        return self

    def GuessOriginalAxisAndDim(self, *args):
        origin = None
        dim = None
        for arg in args:
            if arg.type.type == "INT32":
                origin = arg.value[0]
            else:
                if dim is None:
                    dim = len(arg.type.dimensions)
                else:
                    assert dim == len(arg.type.dimensions)
        assert dim is not None
        origin = dim - 1 if origin is None else origin
        origin = origin + dim if origin < 0 else origin
        return origin, dim

    def AddAxis(self, axis, *args, includeDefault=True, defaultName=None):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        axis = [axis] if type(axis) is int else list(axis)
        var = [AxisConverter(origin, a, dim).Identify(args) for a in axis]
        self.AddVariations(*var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddAllPositiveAxis(self, *args):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        var = [AxisConverter(origin, a, dim).Identify(args) for a in range(dim)]
        self.AddVariations(*var, includeDefault=False)
        return self

    def AddAllAxis(self, *args):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        var = [AxisConverter(origin, a, dim).Identify(args) for a in range(-dim, dim)]
        self.AddVariations(*var, includeDefault=False)
        return self

    def AddDims(self, dims, *args, includeDefault=True, defaultName=None):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        dims = [dims] if type(dims) is int else list(dims)
        drop = list(range(dim))
        drop.pop(origin)
        var = [AxisConverter(origin, origin, dim, drop[0:(dim-i)]).Identify(args) for i in dims]
        self.AddVariations(*var, includeDefault=includeDefault, defaultName=defaultName)
        return self

    def AddAllDims(self, *args):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        drop = list(range(dim))
        drop.pop(origin)
        var = [AxisConverter(origin, origin, dim, drop[0:i]).Identify(args) for i in range(dim)]
        self.AddVariations(*var, includeDefault=False)
        return self

    def AddAllDimsAndPositiveAxis(self, *args):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        var = [AxisConverter(origin, j, dim, range(i)).Identify(args) \
                for i in range(dim) for j in range(i, dim)]
        self.AddVariations(*var, includeDefault=False)
        return self

    def AddAllDimsAndAxis(self, *args):
        origin, dim = self.GuessOriginalAxisAndDim(*args)
        var = [AxisConverter(origin, k, dim, range(i)).Identify(args) \
                for i in range(dim) for j in range(i, dim) for k in [j, j - dim]]
        self.AddVariations(*var, includeDefault=False)
        return self

    def Combine(self, other):
        assert self.model is other.model, "Only examples targetting the same model can be combined"
        assert tuple(self.variations) == tuple(other.variations), \
            "Only examples with the same set of variations can be combined"
        assert self.name == other.name, "Only examples with the same name can be combined"
        self.feedDicts.extend(other.feedDicts)
        return self

    def Dump(self, DumpModel, model_fd, DumpExample, example_fd, DumpTest, test_fd):
        [v.SetToDefaultName() for vs in self.variations for v in vs if v.name is None]
        for variationList in itertools.product(*self.variations):
            # Apply variations
            modelOrigin, feedDictsOrigin = self.model, self.feedDicts
            self.model, self.feedDicts = copy.deepcopy((self.model, self.feedDicts))
            for variation in variationList:
                self.model, self.feedDicts = variation.ApplyTo(self.model, self.feedDicts)
            # Concat names for test and examples
            varNames = [v.name for v in variationList]
            self.testName = NamedTest(FileNames.specName, self.model.name, self.name, *varNames)
            self.examplesName = GlobalVariable("examples", self.model.name, self.name, *varNames)
            if str(self.testName) in Example.versionOverrides:
                self.model.IntroducedIn(Example.versionOverrides[str(self.testName)])
            self.model.WithSuffix(*varNames).Compile()
            # Dump files
            if DumpModel is not None and model_fd is not None:
                DumpModel(self.model, model_fd)
            if DumpExample is not None and example_fd is not None:
                DumpExample(self, example_fd)
            if DumpTest is not None and test_fd is not None:
                DumpTest(self, test_fd)
            # Restore model and feedDicts before variation
            self.model = modelOrigin
            self.feedDicts = feedDictsOrigin
        return self

    # Specifies the RANDOM_MULTINOMIAL distribution tolerance.
    # If set to greater than zero, the input is compared as log-probabilities
    # to the output and must be within this tolerance to pass.
    def WithMultinomialDistributionTolerance(self, expectedTolerance):
      self.expectedMultinomialDistributionTolerance = expectedTolerance
      return self

    # For backward-compatibility with slicing.py
    # Similar to dump_dict, but in python. Used by the slicing tool
    # if referenced is not None, only print operands that are present there
    @staticmethod
    def py_dump_dict(d, referenced):
        ret = []
        for k, v in d.items():
            if referenced != None and k not in referenced:
                continue
            key = str(k)
            init = pprint.pformat(v)
            ret.append("%s: %s" % (key, init))
        return ", ".join(ret)

    # For backward-compatibility with slicing.py
    # similar to dump, but in python. Used by the slicing tool
    # if referenced is not None, only print operands that are present there
    @staticmethod
    def py_dump(example_file, override, referenced):
        Example.CombineAllExamples()
        if len(Example.examples[0].feedDicts) > 0:
            example_no = 0
            example_template = """\
input{no} = {{{inputs}}}
# Only executed during data collection phase
if collecting_data is True:
  Example((input{no}, {{{outputs}}}))
"""
        for i, o in Example.examples[0].feedDicts:
            print ('# Begin of an example', file = example_file)
            inputs = Example.py_dump_dict(i, referenced)
            output_list = []
            for k, v in override.items():
                output_list.append("%s: [0] * %d" % (k, v))
            outputs = ",".join(output_list)

            # TODO: handle >1 outputs
            for k, v in o.items():
                assert k.index == 0
            example_contents = {
                'no': example_no,
                'inputs': inputs,
                'outputs': outputs
            }
            print (example_template.format(**example_contents), file = example_file)

class FileNames:
    specFiles = []
    specNames = []
    modelFiles = []
    exampleFiles = []
    testFiles = []
    specFile = ""
    specName = ""
    modelFile = ""
    exampleFile = ""
    testFile = ""
    ctsFile = ""
    logFile = ""
    version = ""
    fileIndex = 0

    @staticmethod
    def InitializeFileLists(spec, model, example, test, cts="-", log=""):
        # get all spec files and target files
        if os.path.isfile(spec):
            FileNames.specFiles = [os.path.abspath(spec)]
        elif os.path.isdir(spec):
            FileNames.specFiles = sorted([os.path.abspath(os.path.join(spec, f))
                for f in os.listdir(spec) if f.endswith(".mod.py")])
        else:
            assert False, "%s is neither a file or a directory"%spec
        FileNames.specNames = [re.sub(r"\..*", "", os.path.basename(f))
            for f in FileNames.specFiles]
        FileNames.modelFiles = FileNames.ParseTargetFiles(model, ".model.cpp")
        FileNames.exampleFiles = FileNames.ParseTargetFiles(example, ".example.cpp")
        FileNames.testFiles = FileNames.ParseTargetFiles(test, ".mod.py.cpp")
        FileNames.ctsFile = os.path.abspath(cts) if cts != "-" else "-"
        FileNames.logFile = ", \"%s\""%log if log != "" else ""

    @staticmethod
    def ParseTargetFiles(arg, ext):
        numFiles = len(FileNames.specFiles)
        absPath = os.path.abspath(arg)
        if os.path.isdir(arg):
            target = [os.path.join(absPath, f + ext) for f in FileNames.specNames]
        elif arg == "-":
            target = ["-"] * numFiles
        else:
            target = [absPath] * numFiles
        return target

    @staticmethod
    def NextFile():
        if FileNames.fileIndex >= len(FileNames.specFiles):
            return False
        FileNames.specFile = FileNames.specFiles[FileNames.fileIndex]
        FileNames.specName = FileNames.specNames[FileNames.fileIndex]
        FileNames.modelFile = FileNames.modelFiles[FileNames.fileIndex]
        FileNames.exampleFile = FileNames.exampleFiles[FileNames.fileIndex]
        FileNames.testFile = FileNames.testFiles[FileNames.fileIndex]
        FileNames.fileIndex += 1
        NamedObject.existingNames = set()
        NamedVariable.existingNames = set()
        NamedTest.existingNames = set()
        Type.typesMap = dict()
        Model.models = list()
        Example.examples = list()
        Configuration.use_shm_for_weights = False

        # Extract version from absolute file path.
        versionMatch = re.findall(r"/V\d_\d/", FileNames.specFile)
        if len(versionMatch) == 1:
            FileNames.version = versionMatch[0].strip('/')
        else:
            FileNames.version = None
        return True

class Configuration:
    use_shm_for_weights = False
    force_regenerate = False
    test_dynamic_output_shape = True

    @staticmethod
    def useSHM():
        return Configuration.use_shm_for_weights
