#!/usr/bin/env python

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

import os
import sys
import numpy
import flatbuffers
import tflite.Model
import tflite.SubGraph
import tflite.BuiltinOptions
import argparse


# Assume we use only main model in model file
# Get selected operators from file, and return operator index list
def GetOperatorList(oplist_file):
    lines = oplist_file.readlines()
    opcode_list = []

    for line in lines:
        words = line.split()
        for word in words:
            if word.isdigit():
                opcode_list.append(int(word))
            else:
                opcode_range = word.split('-')
                if ((len(opcode_range) == 2) and opcode_range[0].isdigit()
                        and opcode_range[1].isdigit()):
                    start = int(opcode_range[0])
                    end = int(opcode_range[1])
                    for num in range(start, end + 1):
                        opcode_list.append(int(num))
                else:
                    print("Error: Cannot get operator list")
                    print(
                        "Please pass operators as operator index or range list split by space and/or line"
                    )
                    exit(1)

    if len(opcode_list) == 0:
        print("No selected operator")
        exit(1)

    return opcode_list


def GetUsedSubgraphsList(sample_model, subg_num, operator_list, used_subgraphs_list):
    import tflite.IfOptions
    import tflite.WhileOptions

    subg_list = []

    selected_subgraph = sample_model.Subgraphs(subg_num)

    for operator_idx in operator_list:
        selected_operator = selected_subgraph.Operators(operator_idx)
        if selected_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions(
        ).IfOptions:
            selected_builtin_option = selected_operator.BuiltinOptions()
            if_option = tflite.IfOptions.IfOptions()
            if_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

            subg_list.append(if_option.ElseSubgraphIndex())
            subg_list.append(if_option.ThenSubgraphIndex())

        if selected_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions(
        ).WhileOptions:
            selected_builtin_option = selected_operator.BuiltinOptions()
            while_option = tflite.WhileOptions.WhileOptions()
            while_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

            subg_list.append(while_option.BodySubgraphIndex())
            subg_list.append(while_option.CondSubgraphIndex())

    for idx in subg_list:
        if idx not in used_subgraphs_list:
            used_subgraphs_list.append(idx)
            GetUsedSubgraphsList(sample_model, idx,
                                 range(sample_model.Subgraphs(idx).OperatorsLength() - 1),
                                 used_subgraphs_list)


def GenerateOperatorCodes(new_builder, sample_model, used_opcodes_dic,
                          used_subgraphs_dic):
    operator_code_num = sample_model.OperatorCodesLength()
    new_operator_code_list = []
    new_operator_code_string_list = {}

    if operator_code_num == 0:
        return 0

    # Create operator_code string
    for operator_code_idx in range(operator_code_num):
        if operator_code_idx in used_opcodes_dic:
            operator_code = sample_model.OperatorCodes(operator_code_idx)
            operator_code_string = operator_code.CustomCode()
            if operator_code_string and (operator_code_string != "") and (
                    not operator_code_string in new_operator_code_string_list):
                new_operator_code_string_list[
                    operator_code_string] = new_builder.CreateString(operator_code_string)

    # Create tables of operator_code
    for operator_code_idx in range(operator_code_num):
        if operator_code_idx in used_opcodes_dic:
            operator_code = sample_model.OperatorCodes(operator_code_idx)

            # Create operator_code table
            tflite.OperatorCode.OperatorCodeStart(new_builder)
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(new_builder,
                                                           operator_code.BuiltinCode())

            new_operator_code_string = operator_code.CustomCode()
            if new_operator_code_string in new_operator_code_string_list:
                tflite.OperatorCode.OperatorCodeAddCustomCode(
                    new_builder, new_operator_code_string_list[new_operator_code_string])
            new_operator_code = tflite.OperatorCode.OperatorCodeEnd(new_builder)
            new_operator_code_list.append(new_operator_code)

    # Create operator_code vector
    new_operator_code_num = len(new_operator_code_list)
    tflite.Model.ModelStartOperatorCodesVector(new_builder, new_operator_code_num)
    for operator_code_idx in reversed(range(new_operator_code_num)):
        new_builder.PrependUOffsetTRelative(new_operator_code_list[operator_code_idx])

    return new_builder.EndVector(new_operator_code_num)


def GenerateQuantization(new_builder, selected_quantization):
    # Create min vector
    min_num = selected_quantization.MinLength()
    if min_num != 0:
        tflite.QuantizationParameters.QuantizationParametersStartMinVector(
            new_builder, min_num)
        for min_idx in reversed(range(min_num)):
            new_builder.PrependFloat32(selected_quantization.Min(min_idx))
        new_min = new_builder.EndVector(min_num)

    # Create max vector
    max_num = selected_quantization.MaxLength()
    if max_num != 0:
        tflite.QuantizationParameters.QuantizationParametersStartMaxVector(
            new_builder, max_num)
        for max_idx in reversed(range(max_num)):
            new_builder.PrependFloat32(selected_quantization.Max(max_idx))
        new_max = new_builder.EndVector(max_num)

    # Create scale vector
    scale_num = selected_quantization.ScaleLength()
    if scale_num != 0:
        tflite.QuantizationParameters.QuantizationParametersStartScaleVector(
            new_builder, scale_num)
        for scale_idx in reversed(range(scale_num)):
            new_builder.PrependFloat32(selected_quantization.Scale(scale_idx))
        new_scale = new_builder.EndVector(scale_num)

    # Create zero_point vector
    zeropoint_num = selected_quantization.ZeroPointLength()
    if zeropoint_num != 0:
        tflite.QuantizationParameters.QuantizationParametersStartZeroPointVector(
            new_builder, zeropoint_num)
        for zeropoint_idx in reversed(range(zeropoint_num)):
            new_builder.PrependInt64(selected_quantization.ZeroPoint(zeropoint_idx))
        new_zeropoint = new_builder.EndVector(zeropoint_num)

    # Create quantization
    tflite.QuantizationParameters.QuantizationParametersStart(new_builder)
    if min_num != 0:
        tflite.QuantizationParameters.QuantizationParametersAddMin(new_builder, new_min)
    if max_num != 0:
        tflite.QuantizationParameters.QuantizationParametersAddMax(new_builder, new_max)
    if scale_num != 0:
        tflite.QuantizationParameters.QuantizationParametersAddScale(
            new_builder, new_scale)
    if zeropoint_num != 0:
        tflite.QuantizationParameters.QuantizationParametersAddZeroPoint(
            new_builder, new_zeropoint)

    return tflite.QuantizationParameters.QuantizationParametersEnd(new_builder)


def GenerateTensor(new_builder, selected_tensor, used_buffers_dic):

    # Create shape vector for tensor
    shape_num = selected_tensor.ShapeLength()
    tflite.Tensor.TensorStartShapeVector(new_builder, shape_num)
    if shape_num != 0:
        for shape_idx in reversed(range(shape_num)):
            new_builder.PrependInt32(selected_tensor.Shape(shape_idx))
    new_shape = new_builder.EndVector(shape_num)

    # Create tensor_type
    tensor_type = selected_tensor.Type()

    # Create input vector for tensor
    buffer_idx = selected_tensor.Buffer()
    new_buffer_idx = used_buffers_dic[buffer_idx]

    # Create name string
    name_string = selected_tensor.Name()
    if name_string != "":
        new_name = new_builder.CreateString(name_string)

    # Create quantization
    quantization = selected_tensor.Quantization()
    if quantization != None:
        new_quantization = GenerateQuantization(new_builder, quantization)

    # Create tensor
    tflite.Tensor.TensorStart(new_builder)
    tflite.Tensor.TensorAddShape(new_builder, new_shape)
    tflite.Tensor.TensorAddType(new_builder, tensor_type)
    tflite.Tensor.TensorAddBuffer(new_builder, new_buffer_idx)
    if name_string != "":
        tflite.Tensor.TensorAddName(new_builder, new_name)
    if quantization != None:
        tflite.Tensor.TensorAddQuantization(new_builder, new_quantization)

    return tflite.Tensor.TensorEnd(new_builder)


def GenerateTensors(new_builder, selected_subgraph, used_tensors_dic, used_buffers_dic):
    tensor_num = selected_subgraph.TensorsLength()
    new_tensor_list = []

    if tensor_num == 0:
        return 0

    for tensor_idx in range(tensor_num):
        if tensor_idx in used_tensors_dic:
            selected_tensor = selected_subgraph.Tensors(tensor_idx)
            new_tensor = GenerateTensor(new_builder, selected_tensor, used_buffers_dic)
            new_tensor_list.append(new_tensor)

    new_tensor_num = len(new_tensor_list)
    if new_tensor_num == 0:
        return 0

    tflite.SubGraph.SubGraphStartTensorsVector(new_builder, new_tensor_num)
    for new_tensor in reversed(new_tensor_list):
        new_builder.PrependUOffsetTRelative(new_tensor)

    return new_builder.EndVector(new_tensor_num)


def GenerateBuiltinOption(new_builder, selected_builtin_option, builtin_option_type,
                          used_subgraphs_dic):

    # Conv2D option
    import tflite.Conv2DOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions:

        conv2d_options = tflite.Conv2DOptions.Conv2DOptions()
        conv2d_options.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.Conv2DOptions.Conv2DOptionsStart(new_builder)
        tflite.Conv2DOptions.Conv2DOptionsAddPadding(new_builder,
                                                     conv2d_options.Padding())
        tflite.Conv2DOptions.Conv2DOptionsAddStrideW(new_builder,
                                                     conv2d_options.StrideW())
        tflite.Conv2DOptions.Conv2DOptionsAddStrideH(new_builder,
                                                     conv2d_options.StrideH())
        tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(
            new_builder, conv2d_options.DilationWFactor())
        tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(
            new_builder, conv2d_options.DilationHFactor())
        tflite.Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(
            new_builder, conv2d_options.FusedActivationFunction())
        return tflite.Conv2DOptions.Conv2DOptionsEnd(new_builder)

    # DepthwiseConv2D option
    import tflite.DepthwiseConv2DOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).DepthwiseConv2DOptions:

        depthconv2d_option = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        depthconv2d_option.Init(selected_builtin_option.Bytes,
                                selected_builtin_option.Pos)

        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(new_builder)
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(
            new_builder, depthconv2d_option.Padding())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(
            new_builder, depthconv2d_option.StrideW())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(
            new_builder, depthconv2d_option.StrideH())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(
            new_builder, depthconv2d_option.DepthMultiplier())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(
            new_builder, depthconv2d_option.FusedActivationFunction())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(
            new_builder, depthconv2d_option.DilationWFactor())
        tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(
            new_builder, depthconv2d_option.DilationHFactor())
        return tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(new_builder)

    # ConcatEmbeddingsOptions: not supported
    # LSHProjectionOptions: not supported

    # Pool2DPOption
    import tflite.Pool2DOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions:

        pool2d_option = tflite.Pool2DOptions.Pool2DOptions()
        pool2d_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.Pool2DOptions.Pool2DOptionsStart(new_builder)
        tflite.Pool2DOptions.Pool2DOptionsAddPadding(new_builder, pool2d_option.Padding())
        tflite.Pool2DOptions.Pool2DOptionsAddStrideW(new_builder, pool2d_option.StrideW())
        tflite.Pool2DOptions.Pool2DOptionsAddStrideH(new_builder, pool2d_option.StrideH())
        tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(new_builder,
                                                         pool2d_option.FilterWidth())
        tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(new_builder,
                                                          pool2d_option.FilterHeight())
        tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
            new_builder, pool2d_option.FusedActivationFunction())
        return tflite.Pool2DOptions.Pool2DOptionsEnd(new_builder)

    # SVDFOptions: not supported

    # RNNOptions
    import tflite.RNNOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().RNNOptions:

        rnn_option = tflite.RNNOptions.RNNOptions()
        rnn_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.RNNOptions.RNNOptionsStart(new_builder)
        tflite.RNNOptions.RNNOptionsAddFusedActivationFunction(
            new_builder, rnn_option.FusedActivationFunction())
        return tflite.RNNOptions.RNNOptionsEnd(new_builder)

    # FullyConnectedOptions
    import tflite.FullyConnectedOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).FullyConnectedOptions:

        fc_option = tflite.FullyConnectedOptions.FullyConnectedOptions()
        fc_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.FullyConnectedOptions.FullyConnectedOptionsStart(new_builder)
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(
            new_builder, fc_option.FusedActivationFunction())
        return tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(new_builder)

    # SoftmaxOptions
    import tflite.SoftmaxOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SoftmaxOptions:

        softmax_option = tflite.SoftmaxOptions.SoftmaxOptions()
        softmax_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.SoftmaxOptions.SoftmaxOptionsStart(new_builder)
        tflite.SoftmaxOptions.SoftmaxOptionsAddBeta(new_builder, softmax_option.Beta())
        return tflite.SoftmaxOptions.SoftmaxOptionsEnd(new_builder)

    # ConcatenationOptions
    import tflite.ConcatenationOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions:

        concat_option = tflite.ConcatenationOptions.ConcatenationOptions()
        concat_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ConcatenationOptions.ConcatenationOptionsStart(new_builder)
        tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(
            new_builder, concat_option.Axis())
        tflite.ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(
            new_builder, concat_option.FusedActivationFunction())
        return tflite.ConcatenationOptions.ConcatenationOptionsEnd(new_builder)

    # AddOptions
    import tflite.AddOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().AddOptions:

        add_option = tflite.AddOptions.AddOptions()
        add_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.AddOptions.AddOptionsStart(new_builder)
        tflite.AddOptions.AddOptionsAddFusedActivationFunction(
            new_builder, add_option.FusedActivationFunction())
        return tflite.AddOptions.AddOptionsEnd(new_builder)

    # L2NormOptions
    import tflite.L2NormOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().L2NormOptions:

        l2norm_option = tflite.L2NormOptions.L2NormOptions()
        l2norm_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.L2NormOptions.L2NormOptionsStart(new_builder)
        tflite.L2NormOptions.L2NormOptionsAddFusedActivationFunction(
            new_builder, l2norm_option.FusedActivationFunction())
        return tflite.L2NormOptions.L2NormOptionsEnd(new_builder)

    # LocalResponseNormalizationOptions
    import tflite.LocalResponseNormalizationOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).LocalResponseNormalizationOptions:

        lrn_option = tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptions(
        )
        lrn_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsStart(
            new_builder)
        tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddRadius(
            new_builder, lrn_option.Radius())
        tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddBias(
            new_builder, lrn_option.Bias())
        tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddAlpha(
            new_builder, lrn_option.Alpha())
        tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsAddBeta(
            new_builder, lrn_option.Beta())
        return tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsEnd(
            new_builder)

    # LSTMOptions: not supported

    # ResizeBilinearOptions
    import tflite.ResizeBilinearOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).ResizeBilinearOptions:

        resize_bilinear_option = tflite.ResizeBilinearOptions.ResizeBilinearOptions()
        resize_bilinear_option.Init(selected_builtin_option.Bytes,
                                    selected_builtin_option.Pos)

        tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(new_builder)
        tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(
            new_builder, resize_bilinear_option.AlignCorners())
        return tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(new_builder)

    # CallOptions: not supported

    # ReshapeOptions
    import tflite.ReshapeOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions:

        reshape_option = tflite.ReshapeOptions.ReshapeOptions()
        reshape_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        shape_num = reshape_option.NewShapeLength()
        if shape_num != 0:
            tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
                new_builder, shape_num)
            for new_shape_idx in reversed(range(shape_num)):
                new_shape_val = reshape_option.NewShape(new_shape_idx)
                new_builder.PrependInt32(new_shape_val)
            new_shape = new_builder.EndVector(shape_num)

        tflite.ReshapeOptions.ReshapeOptionsStart(new_builder)
        if shape_num != 0:
            tflite.ReshapeOptions.ReshapeOptionsAddNewShape(new_builder, new_shape)
        return tflite.ReshapeOptions.ReshapeOptionsEnd(new_builder)

    # SkipGramOptions: not supported

    # SpaceToDepthOptions
    import tflite.SpaceToDepthOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SpaceToDepthOptions:

        space_to_depth_option = tflite.SpaceToDepthOptions.SpaceToDepthOptions()
        space_to_depth_option.Init(selected_builtin_option.Bytes,
                                   selected_builtin_option.Pos)

        tflite.SpaceToDepthOptions.SpaceToDepthOptionsStart(new_builder)
        tflite.SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(
            new_builder, space_to_depth_option.BlockSize())
        return tflite.SpaceToDepthOptions.SpaceToDepthOptionsEnd(new_builder)

    # EmbeddingLookupSparseOptions: not supported

    # MulOptions
    import tflite.MulOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().MulOptions:

        mul_option = tflite.MulOptions.MulOptions()
        mul_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.MulOptions.MulOptionsStart(new_builder)
        tflite.MulOptions.MulOptionsAddFusedActivationFunction(
            new_builder, mul_option.FusedActivationFunction())
        return tflite.MulOptions.MulOptionsEnd(new_builder)

    # PadOptions
    import tflite.PadOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().PadOptions:

        pad_option = tflite.PadOptions.PadOptions()
        pad_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.PadOptions.PadOptionsStart(new_builder)
        return tflite.PadOptions.PadOptionsEnd(new_builder)

    # GatherOptions
    import tflite.GatherOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().GatherOptions:

        gather_option = tflite.GatherOptions.GatherOptions()
        gather_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.GatherOptions.GatherOptionsStart(new_builder)
        tflite.GatherOptions.GatherOptionsAddAxis(new_builder, gather_option.Axis())
        return tflite.GatherOptions.GatherOptionsEnd(new_builder)

    # BatchToSpaceNDOptions
    import tflite.BatchToSpaceNDOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).BatchToSpaceNDOptions:

        btsnd_option = tflite.BatchToSpaceNDOptions.BatchToSpaceNDOptions()
        btsnd_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.BatchToSpaceNDOptions.BatchToSpaceNDOptionsStart(new_builder)
        return tflite.BatchToSpaceNDOptions.BatchToSpaceNDOptionsEnd(new_builder)

    # SpaceToBatchNDOptions
    import tflite.SpaceToBatchNDOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).SpaceToBatchNDOptions:

        stbnd_option = tflite.SpaceToBatchNDOptions.SpaceToBatchNDOptions()
        stbnd_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.SpaceToBatchNDOptions.SpaceToBatchNDOptionsStart(new_builder)
        return tflite.SpaceToBatchNDOptions.SpaceToBatchNDOptionsEnd(new_builder)

    # TransposeOptions:
    import tflite.TransposeOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().TransposeOptions:

        transpose_option = tflite.TransposeOptions.TransposeOptions()
        transpose_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.TransposeOptions.TransposeOptionsStart(new_builder)
        return tflite.TransposeOptions.TransposeOptionsEnd(new_builder)

    # ReducerOptions
    import tflite.ReducerOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ReducerOptions:

        reducer_option = tflite.ReducerOptions.ReducerOptions()
        reducer_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ReducerOptions.ReducerOptionsStart(new_builder)
        tflite.ReducerOptions.ReducerOptionsAddKeepDims(new_builder,
                                                        reducer_option.KeepDims())
        return tflite.ReducerOptions.ReducerOptionsEnd(new_builder)

    # SubOptions
    import tflite.SubOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SubOptions:

        sub_option = tflite.SubOptions.SubOptions()
        sub_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.SubOptions.SubOptionsStart(new_builder)
        tflite.SubOptions.SubOptionsAddFusedActivationFunction(
            new_builder, sub_option.FusedActivationFunction())
        return tflite.SubOptions.SubOptionsEnd(new_builder)

    # DivOptions
    import tflite.DivOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().DivOptions:

        div_option = tflite.DivOptions.DivOptions()
        div_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.DivOptions.DivOptionsStart(new_builder)
        tflite.DivOptions.DivOptionsAddFusedActivationFunction(
            new_builder, div_option.FusedActivationFunction())
        return tflite.DivOptions.DivOptionsEnd(new_builder)

    # SqueezeOptions
    import tflite.SqueezeOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SqueezeOptions:

        squeeze_option = tflite.SqueezeOptions.SqueezeOptions()
        squeeze_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        squeeze_dims_num = squeeze_option.SqueezeDimsLength()
        if squeeze_dims_num != 0:
            tflite.SqueezeOptions.SqueezeOptionsStartSqueezeDimsVector(
                new_builder, squeeze_dims_num)
            for squeeze_dims_idx in reversed(range(squeeze_dims_num)):
                squeeze_dims_val = squeeze_option.SqueezeDims(squeeze_dims_idx)
                new_builder.PrependInt32(squeeze_dims_val)
            new_squeeze_dims = new_builder.EndVector(squeeze_dims_num)

        tflite.SqueezeOptions.SqueezeOptionsStart(new_builder)
        if squeeze_dims_num != 0:
            tflite.SqueezeOptions.SqueezeOptionsAddSqueezeDims(new_builder,
                                                               new_squeeze_dims)
        return tflite.SqueezeOptions.SqueezeOptionsEnd(new_builder)

    # SequenceRNNOptions: not supported

    # StridedSliceOptions
    import tflite.StridedSliceOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().StridedSliceOptions:

        stride_slice_option = tflite.StridedSliceOptions.StridedSliceOptions()
        stride_slice_option.Init(selected_builtin_option.Bytes,
                                 selected_builtin_option.Pos)

        tflite.StridedSliceOptions.StridedSliceOptionsStart(new_builder)
        tflite.StridedSliceOptions.StridedSliceOptionsAddBeginMask(
            new_builder, stride_slice_option.BeginMask())
        tflite.StridedSliceOptions.StridedSliceOptionsAddEndMask(
            new_builder, stride_slice_option.EndMask())
        tflite.StridedSliceOptions.StridedSliceOptionsAddEllipsisMask(
            new_builder, stride_slice_option.EllipsisMask())
        tflite.StridedSliceOptions.StridedSliceOptionsAddNewAxisMask(
            new_builder, stride_slice_option.NewAxisMask())
        tflite.StridedSliceOptions.StridedSliceOptionsAddShrinkAxisMask(
            new_builder, stride_slice_option.ShrinkAxisMask())

        return tflite.StridedSliceOptions.StridedSliceOptionsEnd(new_builder)

    # ExpOptions
    import tflite.ExpOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ExpOptions:

        exp_option = tflite.ExpOptions.ExpOptions()
        exp_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ExpOptions.ExpOptionsStart(new_builder)
        return tflite.ExpOptions.ExpOptionsEnd(new_builder)

    # TopKV2Options
    import tflite.TopKV2Options
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().TopKV2Options:

        topkv2_option = tflite.TopKV2Options.TopKV2Options()
        topkv2_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.TopKV2Options.TopKV2OptionsStart(new_builder)
        return tflite.TopKV2Options.TopKV2OptionsEnd(new_builder)

    # SplitOptions
    import tflite.SplitOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SplitOptions:

        split_option = tflite.SplitOptions.SplitOptions()
        split_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.SplitOptions.SplitOptionsStart(new_builder)
        tflite.SplitOptions.SplitOptionsAddNumSplits(new_builder,
                                                     split_option.NumSplits())
        return tflite.SplitOptions.SplitOptionsEnd(new_builder)

    # LogSoftmaxOptions: not supported

    # CastOptions: not supported
    import tflite.CastOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().CastOptions:

        cast_option = tflite.CastOptions.CastOptions()
        cast_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.CastOptions.CastOptionsStart(new_builder)
        return tflite.CastOptions.CastOptionsEnd(new_builder)

    # DequantizeOptions:
    import tflite.DequantizeOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().DequantizeOptions:

        dequantize_option = tflite.DequantizeOptions.DequantizeOptions()
        dequantize_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.EqualOptions.DequantizeOptionsStart(new_builder)
        return tflite.DequantizeOptions.DequantizeOptionsEnd(new_builder)

    # MaximumMinimumOptions: not supported

    # ArgMaxOptions
    import tflite.ArgMaxOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ArgMaxOptions:

        arg_max_option = tflite.ArgMaxOptions.ArgMaxOptions()
        arg_max_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ArgMaxOptions.ArgMaxOptionsStart(new_builder)
        tflite.ArgMaxOptions.ArgMaxOptionsAddOutputType(new_builder,
                                                        arg_max_option.OutputType())
        return tflite.ArgMaxOptions.ArgMaxOptionsEnd(new_builder)

    # LessOptions: not supported

    # NegOptions
    import tflite.NegOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().NegOptions:

        neg_option = tflite.NegOptions.NegOptions()
        neg_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.NegOptions.NegOptionsStart(new_builder)
        return tflite.NegOptions.NegOptionsEnd(new_builder)

    # EqualOptions
    import tflite.EqualOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().EqualOptions:

        equal_option = tflite.EqualOptions.EqualOptions()
        equal_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.EqualOptions.EqualOptionsStart(new_builder)
        return tflite.EqualOptions.EqualOptionsEnd(new_builder)

    # PadV2Options: not supported
    # GreaterOptions: not supported
    # GreaterEqualOptions: not supported
    # LessEqualOptions: not supported

    # SelectOptions
    import tflite.SelectOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().SelectOptions:

        select_option = tflite.SelectOptions.SelectOptions()
        select_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.SelectOptions.SelectOptionsStart(new_builder)
        return tflite.SelectOptions.SelectOptionsEnd(new_builder)

    # SliceOptions: not supported

    # TransposeConvOptions
    import tflite.TransposeConvOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().TransposeConvOptions:

        transposeconv_option = tflite.TransposeConvOptions.TransposeConvOptions()
        transposeconv_option.Init(selected_builtin_option.Bytes,
                                  selected_builtin_option.Pos)

        tflite.TransposeConvOptions.TransposeConvOptionsStart(new_builder)
        tflite.TransposeConvOptions.TransposeConvOptionsAddPadding(
            new_builder, transposeconv_option.Padding())
        tflite.TransposeConvOptions.TransposeConvOptionsAddStrideW(
            new_builder, transposeconv_option.StrideW())
        tflite.TransposeConvOptions.TransposeConvOptionsAddStrideH(
            new_builder, transposeconv_option.StrideH())
        return tflite.TransposeConvOptions.TransposeConvOptionsEnd(new_builder)

    # SparseToDenseOptions: not supported
    # TileOptions: not supported

    # ExpandDimsOptions:
    import tflite.ExpandDimsOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ExpandDimsOptions:

        expanddims_option = tflite.ExpandDimsOptions.ExpandDimsOptions()
        expanddims_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ExpandDimsOptions.ExpandDimsOptionsStart(new_builder)
        return tflite.ExpandDimsOptions.ExpandDimsOptionsEnd(new_builder)

    # NotEqualOptions:
    import tflite.NotEqualOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().NotEqualOptions:

        notequal_option = tflite.NotEqualOptions.NotEqualOptions()
        notequal_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.NotEqualOptions.NotEqualOptionsStart(new_builder)
        return tflite.NotEqualOptions.NotEqualOptionsEnd(new_builder)

    # ShapeOptions:
    import tflite.ShapeOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().ShapeOptions:

        shape_option = tflite.ShapeOptions.ShapeOptions()
        shape_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.ShapeOptions.ShapeOptionsStart(new_builder)
        tflite.ShapeOptions.ShapeOptionsAddOutType(new_builder, shape_option.OutType())
        return tflite.ShapeOptions.ShapeOptionsEnd(new_builder)

    # PowOptions: not supported
    # ArgMinOptions: not supported
    # FakeQuantOptions: not supported

    # PackOptions:
    import tflite.PackOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().PackOptions:

        pack_option = tflite.PackOptions.PackOptions()
        pack_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.PackOptions.PackOptionsStart(new_builder)
        tflite.PackOptions.PackOptionsAddValuesCount(new_builder,
                                                     pack_option.ValuesCount())
        tflite.PackOptions.PackOptionsAddAxis(new_builder, pack_option.Axis())
        return tflite.PackOptions.PackOptionsEnd(new_builder)

    # LogicalOrOptions:
    import tflite.LogicalOrOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().LogicalOrOptions:

        logical_or_option = tflite.LogicalAndOptions.LogicalOrOptions()
        logical_or_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.LogicalOrOptions.LogicalOrOptionsStart(new_builder)
        return tflite.LogicalOrOptions.LogicalOrOptionsEnd(new_builder)

    # OneHotOptions: not supported
    import tflite.OneHotOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().OneHotOptions:

        one_hot_option = tflite.OneHotOptions.OneHotOptions()
        one_hot_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.OneHotOptions.OneHotOptionsStart(new_builder)
        tflite.OneHotOptions.OneHotOptionsAddAxis(new_builder, one_hot_option.Axis())
        return tflite.OneHotOptions.OneHotOptionsEnd(new_builder)

    # LogicalNotOptions
    import tflite.LogicalNotOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().LogicalNotOptions:

        equal_option = tflite.LogicalNotOptions.LogicalNotOptions()
        equal_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.LogicalNotOptions.LogicalNotOptionsStart(new_builder)
        return tflite.LogicalNotOptions.LogicalNotOptionsEnd(new_builder)

    # UnpackOptions:
    import tflite.UnpackOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().UnpackOptions:

        unpack_option = tflite.UnpackOptions.UnpackOptions()
        unpack_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.UnpackOptions.UnpackOptionsStart(new_builder)
        tflite.UnpackOptions.UnpackOptionsAddNum(new_builder, unpack_option.Num())
        tflite.UnpackOptions.UnpackOptionsAddAxis(new_builder, unpack_option.Axis())
        return tflite.UnpackOptions.UnpackOptionsEnd(new_builder)

    # FloorDivOptions: not supported
    # SquareOptions: not supported
    # ZerosLikeOptions: not supported
    # FillOptions: not supported

    # LogicalAndOptions
    import tflite.LogicalAndOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().LogicalAndOptions:

        logical_and_option = tflite.LogicalAndOptions.LogicalAndOptions()
        logical_and_option.Init(selected_builtin_option.Bytes,
                                selected_builtin_option.Pos)

        tflite.LogicalAndOptions.LogicalAndOptionsStart(new_builder)
        return tflite.LogicalAndOptions.LogicalAndOptionsEnd(new_builder)

    # LogicalNotOptions: not supported
    # UnpackOptions: not supported
    # FloorDivOptions: not supported
    # SquareOptions: not supported
    # ZerosLikeOptions: not supported
    # FillOptions: not supported
    # BidirectionalSequenceLSTMOptions: not supported
    # BidirectionalSequenceRNNOptions: not supported
    # UnidirectionalSequenceLSTMOptions: not supported
    # FloorModOptions: not supported
    # RangeOptions: not supported
    # ResizeNearestNeighborOptions: not supported

    # LeakyReluOptions
    import tflite.LeakyReluOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().LeakyReluOptions:

        leaky_relu_option = tflite.LeakyReluOptions.LeakyReluOptions()
        leaky_relu_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.LeakyReluOptions.LeakyReluOptionsStart(new_builder)
        tflite.LeakyReluOptions.LeakyReluOptionsAddAlpha(new_builder,
                                                         leaky_relu_option.Alpha())
        return tflite.LeakyReluOptions.LeakyReluOptionsEnd(new_builder)

    # SquaredDifferenceOptions
    import tflite.SquaredDifferenceOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions(
    ).SquaredDifferenceOptions:

        squared_difference_option = tflite.SquaredDifferenceOptions.SquaredDifferenceOptions(
        )
        squared_difference_option.Init(selected_builtin_option.Bytes,
                                       selected_builtin_option.Pos)

        tflite.SquaredDifferenceOptions.SquaredDifferenceOptionsStart(new_builder)
        return tflite.SquaredDifferenceOptions.SquaredDifferenceOptionsEnd(new_builder)

    # MirrorPadOptions: not supported
    # AbsOptions: not supported
    # SplitVOptions: not supported

    # IfOptions
    import tflite.IfOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().IfOptions:

        if_option = tflite.IfOptions.IfOptions()
        if_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.IfOptions.IfOptionsStart(new_builder)
        tflite.IfOptions.IfOptionsAddElseSubgraphIndex(
            new_builder, used_subgraphs_dic[if_option.ElseSubgraphIndex()])
        tflite.IfOptions.IfOptionsAddThenSubgraphIndex(
            new_builder, used_subgraphs_dic[if_option.ThenSubgraphIndex()])
        return tflite.IfOptions.IfOptionsEnd(new_builder)

    # WhileOptions
    import tflite.WhileOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().WhileOptions:

        while_option = tflite.WhileOptions.WhileOptions()
        while_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.WhileOptions.WhileOptionsStart(new_builder)
        tflite.WhileOptions.WhileOptionsAddBodySubgraphIndex(
            new_builder, used_subgraphs_dic[while_option.BodySubgraphIndex()])
        tflite.WhileOptions.WhileOptionsAddCondSubgraphIndex(
            new_builder, used_subgraphs_dic[while_option.CondSubgraphIndex()])
        return tflite.WhileOptions.WhileOptionsEnd(new_builder)

    # BCQFullyConnectedOptions
    import tflite.BCQFullyConnectedOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().BCQFullyConnectedOptions:

        bcqfc_option = tflite.BCQFullyConnectedOptions.BCQFullyConnectedOptions()
        bcqfc_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.BCQFullyConnectedOptions.BCQFullyConnectedOptionsStart(new_builder)
        tflite.BCQFullyConnectedOptions.BCQFullyConnectedOptionsAddWeightsHiddenSize(new_builder, bcqfc_option.WeightsHiddenSize())
        tflite.BCQFullyConnectedOptions.BCQFullyConnectedOptionsAddFusedActivationFunction(new_builder, bcqfc_option.FusedActivationFunction())
        return tflite.BCQFullyConnectedOptions.BCQFullyConnectedOptionsEnd(new_builder)

    # BCQGatherOptions
    import tflite.BCQGatherOptions
    if builtin_option_type == tflite.BuiltinOptions.BuiltinOptions().BCQGatherOptions:

        bcqgather_option = tflite.BCQGatherOptions.BCQGatherOptions()
        bcqgather_option.Init(selected_builtin_option.Bytes, selected_builtin_option.Pos)

        tflite.BCQGatherOptions.BCQGatherOptionsStart(new_builder)
        tflite.BCQGatherOptions.BCQGatherOptionsAddAxis(new_builder, bcqgather_option.Axis())
        tflite.BCQGatherOptions.BCQGatherOptionsAddInputHiddenSize(new_builder, bcqgather_option.InputHiddenSize())
        return tflite.BCQGatherOptions.BCQGatherOptionsEnd(new_builder)

    # Cannot handle builtin option type yet
    print("Cannot handle BuiltinOptions {} yet. See BuiltinOptions.py for op name".format(
        builtin_option_type))
    exit(1)


def GenerateOperator(new_builder, selected_operator, used_tensors_dic, used_opcodes_dic,
                     used_subgraphs_dic):

    # define opcode_index
    opcode_index = selected_operator.OpcodeIndex()
    new_opcode_index = used_opcodes_dic[opcode_index]

    # create input vector
    input_num = selected_operator.InputsLength()
    if input_num != 0:
        tflite.Operator.OperatorStartInputsVector(new_builder, input_num)
        for input_idx in reversed(range(input_num)):
            input_tensor_idx = selected_operator.Inputs(input_idx)
            if input_tensor_idx == -1:
                new_input_tensor_idx = -1
            else:
                new_input_tensor_idx = used_tensors_dic[input_tensor_idx]
            new_builder.PrependInt32(new_input_tensor_idx)
        new_input = new_builder.EndVector(input_num)

    # create output_vector
    output_num = selected_operator.OutputsLength()
    if output_num != 0:
        tflite.Operator.OperatorStartOutputsVector(new_builder, output_num)
        for output_idx in reversed(range(output_num)):
            output_tensor_idx = selected_operator.Outputs(output_idx)
            new_output_tensor_idx = used_tensors_dic[output_tensor_idx]
            new_builder.PrependInt32(new_output_tensor_idx)
        new_output = new_builder.EndVector(output_num)

    # Create builtin_option
    builtin_option_type = selected_operator.BuiltinOptionsType()
    if builtin_option_type != 0:
        selected_builtin_option = selected_operator.BuiltinOptions()
        new_builtin_option = GenerateBuiltinOption(
            new_builder, selected_builtin_option, builtin_option_type, used_subgraphs_dic)

    # Create custum option vector
    custom_option_num = selected_operator.CustomOptionsLength()
    if custom_option_num != 0:
        tflite.Operator.OperatorStartCustomOptionsVector(new_builder, custom_option_num)
        for custom_option_idx in reversed(range(custom_option_num)):
            new_builder.PrependUint8(selected_operator.CustomOptions(custom_option_idx))
        new_custom_option = new_builder.EndVector(custom_option_num)

    # Create custum option type
    custom_option_type = selected_operator.CustomOptionsFormat()

    # Create operator
    tflite.Operator.OperatorStart(new_builder)
    tflite.Operator.OperatorAddOpcodeIndex(new_builder, new_opcode_index)
    if input_num != 0:
        tflite.Operator.OperatorAddInputs(new_builder, new_input)
    if output_num != 0:
        tflite.Operator.OperatorAddOutputs(new_builder, new_output)
    tflite.Operator.OperatorAddBuiltinOptionsType(new_builder, builtin_option_type)
    if builtin_option_type != 0:
        tflite.Operator.OperatorAddBuiltinOptions(new_builder, new_builtin_option)
    if custom_option_num != 0:
        tflite.Operator.OperatorAddCustomOptions(new_builder, new_custom_option)
    tflite.Operator.OperatorAddCustomOptionsFormat(new_builder, custom_option_type)
    return tflite.Operator.OperatorEnd(new_builder)


def GenerateOperators(new_builder, selected_subgraph, operator_list, used_tensors_dic,
                      used_opcodes_dic, used_subgraphs_dic):
    operator_num = selected_subgraph.OperatorsLength()
    new_operator_list = []

    if operator_num == 0:
        return 0

    for operator_idx in range(operator_num):
        if operator_idx in operator_list:
            selected_operator = selected_subgraph.Operators(operator_idx)
            new_operator = GenerateOperator(new_builder, selected_operator,
                                            used_tensors_dic, used_opcodes_dic,
                                            used_subgraphs_dic)
            new_operator_list.append(new_operator)

    new_operator_num = len(new_operator_list)
    if new_operator_num == 0:
        return 0

    tflite.SubGraph.SubGraphStartOperatorsVector(new_builder, new_operator_num)
    for new_operator in reversed(new_operator_list):
        new_builder.PrependUOffsetTRelative(new_operator)

    return new_builder.EndVector(new_operator_num)


def GenerateSubgraph(new_builder, selected_subgraph, operator_list, new_input_tensor,
                     new_output_tensor, used_tensors_dic, used_buffers_dic,
                     used_opcodes_dic, used_subgraphs_dic):

    # Tensors
    tensors = GenerateTensors(new_builder, selected_subgraph, used_tensors_dic,
                              used_buffers_dic)

    # Create input vector for subgraph table
    new_input_tensor_num = len(new_input_tensor)
    if new_input_tensor_num != 0:
        tflite.SubGraph.SubGraphStartInputsVector(new_builder, new_input_tensor_num)
        for input_tensor_idx in reversed(new_input_tensor):
            new_input_tensor_idx = used_tensors_dic[input_tensor_idx]
            new_builder.PrependInt32(new_input_tensor_idx)
        new_inputs = new_builder.EndVector(new_input_tensor_num)

    # Create output vector for subgraph table
    new_output_tensor_num = len(new_output_tensor)
    if new_output_tensor_num != 0:
        tflite.SubGraph.SubGraphStartOutputsVector(new_builder, new_output_tensor_num)
        for output_tensor_idx in reversed(new_output_tensor):
            new_output_tensor_idx = used_tensors_dic[output_tensor_idx]
            new_builder.PrependInt32(new_output_tensor_idx)
        new_outputs = new_builder.EndVector(new_output_tensor_num)

    # Operators
    operators = GenerateOperators(new_builder, selected_subgraph, operator_list,
                                  used_tensors_dic, used_opcodes_dic, used_subgraphs_dic)

    # Name
    subgraph_name = selected_subgraph.Name()
    have_name = False
    if subgraph_name and subgraph_name != "":
        have_name = True
        new_subgraph_name = new_builder.CreateString(subgraph_name)

    tflite.SubGraph.SubGraphStart(new_builder)
    tflite.SubGraph.SubGraphAddTensors(new_builder, tensors)
    if new_input_tensor_num != 0:
        tflite.SubGraph.SubGraphAddInputs(new_builder, new_inputs)
    if new_output_tensor_num != 0:
        tflite.SubGraph.SubGraphAddOutputs(new_builder, new_outputs)
    tflite.SubGraph.SubGraphAddOperators(new_builder, operators)
    if have_name:
        tflite.SubGraph.SubGraphAddName(new_builder, new_subgraph_name)

    return tflite.SubGraph.SubGraphEnd(new_builder)


def GenerateSubgraphs(args, new_builder, sample_model, operator_list, new_input_tensor,
                      new_output_tensor, used_tensors_dic, used_buffers_dic,
                      used_opcodes_dic, used_subgraphs_dic):

    new_subgraph_list = []

    # The selected subgraph will be primary subgraph of the model to be created newly
    selected_subgraph = sample_model.Subgraphs(args.subgraph)

    # k: old subg index, v: new subg index
    # new subg index is sequential in used_subgraphs_dic
    for k, v in used_subgraphs_dic.items():
        print("Append subgraphs, old index : ", k, ", new index : ", v)
        if k == args.subgraph:
            assert v == 0
            new_subgraph = GenerateSubgraph(new_builder, selected_subgraph, operator_list,
                                            new_input_tensor, new_output_tensor,
                                            used_tensors_dic, used_buffers_dic,
                                            used_opcodes_dic, used_subgraphs_dic)
            new_subgraph_list.append(new_subgraph)
        else:
            subg = sample_model.Subgraphs(k)
            subg_opperator_idx_list = range(subg.OperatorsLength())
            subg_input_tensors = subg.InputsAsNumpy()
            subg_output_tensors = subg.OutputsAsNumpy()
            subg_tensors = range(subg.TensorsLength())
            subg_tensors_dic = {tensor_idx: tensor_idx for tensor_idx in subg_tensors}
            subg_buffers_dic = {(subg.Tensors(idx)).Buffer():
                                (subg.Tensors(idx)).Buffer()
                                for idx in subg_tensors}
            new_subgraph = GenerateSubgraph(new_builder, subg, subg_opperator_idx_list,
                                            subg_input_tensors, subg_output_tensors,
                                            subg_tensors_dic, subg_buffers_dic,
                                            used_opcodes_dic, used_subgraphs_dic)
            new_subgraph_list.append(new_subgraph)

    new_subgraph_num = len(new_subgraph_list)
    tflite.Model.ModelStartSubgraphsVector(new_builder, new_subgraph_num)
    for subgraph_idx in reversed(range(new_subgraph_num)):
        new_builder.PrependUOffsetTRelative(new_subgraph_list[subgraph_idx])

    return new_builder.EndVector(new_subgraph_num)


def GenerateBuffers(new_builder, sample_model, used_buffers_dic):
    buffer_num = sample_model.BuffersLength()
    new_buffer_data_list = {}
    new_buffer_list = []

    if buffer_num == 0:
        return 0

    # Create data vector for buffer table
    for buffer_idx in range(buffer_num):
        buffer = sample_model.Buffers(buffer_idx)
        buffer_length = buffer.DataLength()

        if (buffer_length != 0) and (buffer_idx in used_buffers_dic):
            tflite.Buffer.BufferStartDataVector(new_builder, buffer_length)
            for buffer_data_idx in reversed(range(buffer_length)):
                new_builder.PrependUint8(buffer.Data(buffer_data_idx))
            new_buffer = new_builder.EndVector(buffer_length)
            new_buffer_data_list[buffer_idx] = new_buffer

    # Create tables of buffer
    for buffer_idx in range(buffer_num):
        buffer = sample_model.Buffers(buffer_idx)

        if buffer_idx in used_buffers_dic:
            # Create buffer table
            tflite.Buffer.BufferStart(new_builder)
            if buffer.DataLength() != 0:
                tflite.Buffer.BufferAddData(new_builder, new_buffer_data_list[buffer_idx])
            new_buffer = tflite.Buffer.BufferEnd(new_builder)
            new_buffer_list.append(new_buffer)

    # Create buffer vector
    new_buffer_num = len(new_buffer_list)
    if new_buffer_num == 0:
        return 0

    tflite.Model.ModelStartBuffersVector(new_builder, new_buffer_num)
    for new_buffer_idx in reversed(range(new_buffer_num)):
        new_builder.PrependUOffsetTRelative(new_buffer_list[new_buffer_idx])

    return new_builder.EndVector(new_buffer_num)


def GenerateModel(args, new_builder, sample_model, operator_list, new_input_tensors,
                  new_output_tensors, used_tensors_dic, used_buffers_dic,
                  used_opcodes_dic, used_subgraphs_dic):
    # uint
    version = sample_model.Version()

    # pointer of operator code 'table' vector
    operator_codes = GenerateOperatorCodes(new_builder, sample_model, used_opcodes_dic,
                                           used_subgraphs_dic)

    # subgraphs
    subgraphs = GenerateSubgraphs(args, new_builder, sample_model, operator_list,
                                  new_input_tensors, new_output_tensors, used_tensors_dic,
                                  used_buffers_dic, used_opcodes_dic, used_subgraphs_dic)

    # description
    description_string = new_builder.CreateString(sample_model.Description())

    # buffers
    buffers = GenerateBuffers(new_builder, sample_model, used_buffers_dic)

    # Generate model
    tflite.Model.ModelStart(new_builder)
    tflite.Model.ModelAddVersion(new_builder, version)
    tflite.Model.ModelAddOperatorCodes(new_builder, operator_codes)
    tflite.Model.ModelAddSubgraphs(new_builder, subgraphs)
    tflite.Model.ModelAddDescription(new_builder, description_string)
    tflite.Model.ModelAddBuffers(new_builder, buffers)

    return tflite.Model.ModelEnd(new_builder)


def main(args):
    input_model_file = args.input_model
    oplist_file = args.opcode_list
    output_model_file = args.output_model
    subgraph = args.subgraph

    # Parse operator list file
    operator_list = GetOperatorList(oplist_file)

    # Get sample model and subgraph
    # We use only 1st subgraph
    sample_buf = input_model_file.read()
    sample_buf = bytearray(sample_buf)
    sample_model = tflite.Model.Model.GetRootAsModel(sample_buf, 0)
    sample_subgraph = sample_model.Subgraphs(subgraph)

    used_subgraphs_list = []
    used_subgraphs_list.append(args.subgraph)
    GetUsedSubgraphsList(sample_model, args.subgraph, operator_list, used_subgraphs_list)

    used_subgraphs_dic = {}
    for new_subgraph_idx in range(len(used_subgraphs_list)):
        sample_subgraph_idx = used_subgraphs_list[new_subgraph_idx]
        used_subgraphs_dic[sample_subgraph_idx] = new_subgraph_idx

    # Collect used tensor & used operator
    used_tensors = []
    used_opcodes = []

    for operator_idx in operator_list:
        operator = sample_subgraph.Operators(operator_idx)
        for input_idx in range(operator.InputsLength()):
            input_tensor_idx = operator.Inputs(input_idx)
            if not input_tensor_idx == -1 and not input_tensor_idx in used_tensors:
                # default: same as input sample
                used_tensors.append(input_tensor_idx)

        for output_idx in range(operator.OutputsLength()):
            output_tensor_idx = operator.Outputs(output_idx)
            if not output_tensor_idx in used_tensors:
                # default: same as input sample
                used_tensors.append(output_tensor_idx)

        opcode_idx = operator.OpcodeIndex()
        if not opcode_idx in used_opcodes:
            used_opcodes.append(opcode_idx)

    # Append opcodes of child subgraphs
    for subgraph_idx in used_subgraphs_list:
        if subgraph_idx == subgraph:
            continue
        for operator_idx in range(sample_model.Subgraphs(subgraph_idx).OperatorsLength()):
            operator = sample_model.Subgraphs(subgraph_idx).Operators(operator_idx)
            opcode_idx = operator.OpcodeIndex()
            if not opcode_idx in used_opcodes:
                used_opcodes.append(opcode_idx)

    used_tensors.sort()
    used_opcodes.sort()

    # Collect used buffer
    # buffer[0] should be blank. So it should start from 1
    used_buffers = [0]

    for used_tensor in used_tensors:
        # key and value is same in prepare phase
        buf_idx = (sample_subgraph.Tensors(used_tensor)).Buffer()
        used_buffers.append(buf_idx)

    # Append buffers of tensors of child subgraphs
    for subgraph_idx in used_subgraphs_list:
        if subgraph_idx == subgraph:
            continue
        for tensor_idx in range(sample_model.Subgraphs(subgraph_idx).TensorsLength()):
            tensor = sample_model.Subgraphs(subgraph_idx).Tensors(tensor_idx)
            used_buffers.append(tensor.Buffer())

    used_buffers.sort()

    # Assign new index for operator
    used_opcodes_dic = {}

    for new_operator_idx in range(len(used_opcodes)):
        sample_operator_idx = used_opcodes[new_operator_idx]
        used_opcodes_dic[sample_operator_idx] = new_operator_idx

    # Assign new index for tensor
    used_tensors_dic = {}

    for new_tensor_idx in range(len(used_tensors)):
        sample_tensor_idx = used_tensors[new_tensor_idx]
        used_tensors_dic[sample_tensor_idx] = new_tensor_idx

    # Assign new index for buffer
    used_buffers_dic = {}

    for new_buffer_idx in range(len(used_buffers)):
        sample_buffer_idx = used_buffers[new_buffer_idx]
        used_buffers_dic[sample_buffer_idx] = new_buffer_idx

    # Find input & output tensor in new model
    new_input_tensors = used_tensors[:]
    new_output_tensors = used_tensors[:]

    for operator_idx in operator_list:
        operator = sample_subgraph.Operators(operator_idx)
        for input_idx in range(operator.InputsLength()):
            input_tensor_idx = operator.Inputs(input_idx)
            if input_tensor_idx == -1:
                continue
            if input_tensor_idx in new_output_tensors:
                new_output_tensors.remove(input_tensor_idx)
            if input_tensor_idx in new_input_tensors:
                matched_buffer_idx = sample_subgraph.Tensors(input_tensor_idx).Buffer()
                matched_buffer = sample_model.Buffers(matched_buffer_idx)
                if matched_buffer.DataLength() != 0:
                    new_input_tensors.remove(input_tensor_idx)

        for output_idx in range(operator.OutputsLength()):
            output_tensor_idx = operator.Outputs(output_idx)
            if output_tensor_idx in new_input_tensors:
                new_input_tensors.remove(output_tensor_idx)
            if output_tensor_idx in new_output_tensors:
                matched_buffer_idx = sample_subgraph.Tensors(output_tensor_idx).Buffer()
                matched_buffer = sample_model.Buffers(matched_buffer_idx)
                if matched_buffer.DataLength() != 0:
                    new_output_tensors.remove(input_tensor_idx)

    new_input_tensors_newidx = []
    new_output_tensors_newidx = []

    for input_tensor_idx in new_input_tensors:
        new_input_tensors_newidx.append(used_tensors_dic[input_tensor_idx])
    for output_tensor_idx in new_output_tensors:
        new_output_tensors_newidx.append(used_tensors_dic[output_tensor_idx])

    print("Input tensor(s): " + str(new_input_tensors_newidx))
    print("Output tensor(s): " + str(new_output_tensors_newidx))

    # Create new model file
    new_builder = flatbuffers.Builder(1024)

    new_model = GenerateModel(args, new_builder, sample_model, operator_list,
                              new_input_tensors, new_output_tensors, used_tensors_dic,
                              used_buffers_dic, used_opcodes_dic, used_subgraphs_dic)

    new_builder.Finish(new_model, file_identifier=b'TFL3')
    new_buf = new_builder.Output()

    output_model_file.write(new_buf)


if __name__ == '__main__':
    # Define argument and read
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "input_model",
        type=argparse.FileType('rb'),
        help="input tflite model file to read")
    arg_parser.add_argument(
        "opcode_list",
        type=argparse.FileType('r'),
        help="text file including selected operator list")
    arg_parser.add_argument(
        "output_model", type=argparse.FileType('wb'), help="output tflite model file")
    arg_parser.add_argument(
        '-g', '--subgraph', type=int, default=0, help="subgraph to use (default: 0)")

    # TODO
    #   Select multiple subgraph
    #   Select subgraph by using opcode list file
    #   Select opcode list by using argument

    args = arg_parser.parse_args()

    # Call main function
    main(args)
