/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "open_cl/kernels/ConvBuffer1x1.h"

#include <array>
#include <string>
#include <utility>

#include "open_cl/ClDevice.h"
#include "open_cl/kernels/Util.h"
#include "open_cl/kernels/WorkGroupPicking.h"
#include "open_cl/Precision.h"
#include "open_cl/TensorType.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{

// element_size must be 1, 2 or 4
// 1 - is FLT4
// 2 - is FLT8
// 4 - is FLT16
// This function generates code for arithmetic part of convolution
std::string GetComputationPart(const int3 &block_size, int element_size,
                               CalculationsPrecision precision)
{
  const std::string hexes[16] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                 "8", "9", "a", "b", "c", "d", "e", "f"};
  std::string c;
  for (int z = 0; z < block_size.z; ++z)
  {
    const std::string z_s = std::to_string(z);
    c += "    FLT16 W" + z_s + " = weights_cache[" + z_s + "];\n";
    for (int y = 0; y < block_size.y; ++y)
    {
      for (int x = 0; x < block_size.x; ++x)
      {
        std::string s_index = std::to_string(y * block_size.x + x);
        for (int e = 0; e < element_size; ++e)
        {
          std::string r_index = z_s + std::to_string(y) + std::to_string(x * element_size + e);
          const std::string f0 = "W" + z_s + ".s0123";
          const std::string f1 = "W" + z_s + ".s4567";
          const std::string f2 = "W" + z_s + ".s89ab";
          const std::string f3 = "W" + z_s + ".scdef";
          switch (precision)
          {
            case CalculationsPrecision::F32:
            case CalculationsPrecision::F16:
              c += "    r" + r_index + " += " + f0 + " * s" + s_index + ".s" + hexes[e * 4 + 0] +
                   ";\n";
              c += "    r" + r_index + " += " + f1 + " * s" + s_index + ".s" + hexes[e * 4 + 1] +
                   ";\n";
              c += "    r" + r_index + " += " + f2 + " * s" + s_index + ".s" + hexes[e * 4 + 2] +
                   ";\n";
              c += "    r" + r_index + " += " + f3 + " * s" + s_index + ".s" + hexes[e * 4 + 3] +
                   ";\n";
              break;
            case CalculationsPrecision::F32_F16:
              c += "    r" + r_index + " += convert_float4(" + f0 + " * s" + s_index + ".s" +
                   hexes[e * 4 + 0] + " + " + f1 + " * s" + s_index + ".s" + hexes[e * 4 + 1] +
                   " + " + f2 + " * s" + s_index + ".s" + hexes[e * 4 + 2] + " + " + f3 + " * s" +
                   s_index + ".s" + hexes[e * 4 + 3] + ");\n";
              break;
          }
        }
      }
    }
  }
  return c;
}

ConvBuffer1x1::ConvParams GetBestParams(const DeviceInfo &device_info,
                                        const OperationDef &definition, const BHWC &shape, int,
                                        int dst_depth)
{
  ConvBuffer1x1::ConvParams conv_params;
  conv_params.element_size = 4;
  conv_params.block_size = int3(1, 1, 1);
  if (!device_info.IsMali())
  {
    return conv_params;
  }
  bool can_use_flt8 =
    (shape.w * shape.b) % 2 == 0 && definition.precision != CalculationsPrecision::F32;
  bool is_midgard = device_info.IsMali() && device_info.mali_info.IsMidgard();
  if (is_midgard)
  {
    if (can_use_flt8)
    {
      conv_params.element_size = 8;
    }
    if (definition.precision == CalculationsPrecision::F16 || !can_use_flt8)
    {
      conv_params.block_size.x = 2;
    }
    return conv_params;
  }

  int task_size = shape.w * shape.b * shape.h * dst_depth;
  int block_size = GetRecommendedBlockSizeForConv(device_info, definition.precision, task_size);

  if (!can_use_flt8 && block_size > 4)
  {
    block_size = 4;
  }

  if (can_use_flt8 && block_size >= 2)
  {
    conv_params.element_size = 8;
    block_size /= 2;
  }
  if (block_size == 4)
  {
    conv_params.block_size.x = 2;
    if (definition.precision == CalculationsPrecision::F32 && dst_depth < 32)
    {
      conv_params.block_size.y = 2;
    }
    else
    {
      conv_params.block_size.z = 2;
    }
  }
  else if (block_size == 2)
  {
    if (dst_depth >= 32)
    {
      conv_params.block_size.z = 2;
    }
    else
    {
      conv_params.block_size.x = 2;
    }
  }

  return conv_params;
}

ConvBuffer1x1::ConvParams GetBestParams(const DeviceInfo &device_info,
                                        const OperationDef &definition, int, int)
{
  ConvBuffer1x1::ConvParams conv_params;
  conv_params.element_size = 4;
  conv_params.block_size = int3(1, 1, 1);
  if (device_info.IsMali() && definition.precision == CalculationsPrecision::F16 &&
      device_info.compute_units_count <= 4)
  {
    conv_params.block_size.x *= 2;
  }
  return conv_params;
}

} // namespace

ConvBuffer1x1::ConvBuffer1x1(const OperationDef &definition, const ConvParams &conv_params)
  : GPUOperation(definition), conv_params_(conv_params)
{
  code_ = GenerateConvBuffer1x1(definition_, conv_params_, &args_);
  work_group_size_ = int3(2, 4, 1);
}

ConvBuffer1x1::ConvBuffer1x1(ConvBuffer1x1 &&operation)
  : GPUOperation(std::move(operation)), conv_params_(std::move(operation.conv_params_))
{
}

ConvBuffer1x1 &ConvBuffer1x1::operator=(ConvBuffer1x1 &&operation)
{
  if (this != &operation)
  {
    std::swap(conv_params_, operation.conv_params_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ConvBuffer1x1::GenerateConvBuffer1x1(const OperationDef &op_def,
                                                 const ConvBuffer1x1::ConvParams &conv_params,
                                                 Arguments *)
{
  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported())
  {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  if (conv_params_.element_size == 8)
  {
    src_desc.SetStateVar("ElementsX2", "true");
  }
  else if (conv_params_.element_size == 16)
  {
    src_desc.SetStateVar("ElementsX4", "true");
  }
  AddSrcTensor("src_tensor", src_desc);
  if (op_def.src_tensors.size() == 2)
  {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].data_type;
    desc.element_size = 16;
    desc.memory_type = MemoryType::GLOBAL;
    AddSrcBuffer("weights", desc);
  }

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported())
  {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);

  std::string c = GetCommonDefines(op_def.precision);
  switch (op_def.precision)
  {
    case CalculationsPrecision::F32:
      c += "#define FLT8 float8\n";
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT8 half8\n";
      c += "#define FLT16 half16\n";
      break;
  }

  const int3 block_size = conv_params.block_size;
  const int element_size = conv_params.element_size / 4;

  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0) * " + std::to_string(block_size.x * element_size) + ";\n";
  c += "  int X_SRC = get_global_id(0) * " + std::to_string(block_size.x) + ";\n";
  c += "  int Y = get_global_id(1) * " + std::to_string(block_size.y) + ";\n";
  c += "  int Z = get_global_id(2) * " + std::to_string(block_size.z) + ";\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) return;\n";
  if (conv_params.different_weights_for_height)
  {
    c += "  __global FLT16* weights_cache = args.weights.GetPtr() + (Z * "
         "args.src_tensor.Height() + "
         "Y * " +
         std::to_string(block_size.z) +
         ") * "
         "args.src_tensor.Slices();\n";
  }
  else
  {
    c += "  __global FLT16* weights_cache = args.weights.GetPtr() + Z * "
         "args.src_tensor.Slices();\n";
  }
  for (int z = 0; z < block_size.z; ++z)
  {
    const std::string z_s = std::to_string(z);
    c += "  ACCUM_FLT4 bias_val_" + z_s + " = TO_ACCUM_TYPE(args.biases.Read(Z + " + z_s + "));\n";
    for (int y = 0; y < block_size.y; ++y)
    {
      for (int x = 0; x < block_size.x * element_size; ++x)
      {
        c += "  ACCUM_FLT4 r" + z_s + std::to_string(y) + std::to_string(x) + " = bias_val_" + z_s +
             ";\n";
      }
    }
  }
  for (int x = 0; x < block_size.x; ++x)
  {
    std::string x_s = std::to_string(x);
    c += "  int xc" + x_s + " = min(X_SRC + " + std::to_string(x) +
         ", args.src_tensor.Width() - 1);\n";
  }
  for (int y = 0; y < block_size.y; ++y)
  {
    std::string y_s = std::to_string(y);
    c += "  int yc" + y_s + " = min(Y + " + y_s + ", args.src_tensor.Height() - 1);\n";
  }
  for (int y = 0; y < block_size.y; ++y)
  {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x)
    {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * block_size.x + x);
      c += "  int src_addr_" + i_s + " = (yc" + y_s + ") * args.src_tensor.Width() + (xc" + x_s +
           ");\n";
    }
  }
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  for (int y = 0; y < block_size.y; ++y)
  {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x)
    {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * block_size.x + x);
      c += "    FLT" + std::to_string(element_size * 4) + " s" + i_s +
           " = args.src_tensor.Read(src_addr_" + i_s + ");\n";
    }
  }
  c += GetComputationPart(block_size, element_size, op_def.precision);
  for (int i = 0; i < block_size.x * block_size.y; ++i)
  {
    std::string i_s = std::to_string(i);
    c += "    src_addr_" + i_s + " += args.src_tensor.SliceStride();\n";
  }
  c += "    weights_cache += " + std::to_string(block_size.z) + ";\n";
  c += "  }\n"; // SRC_SLICES

  for (int z = 0; z < block_size.z; ++z)
  {
    const std::string z_s = std::to_string(z);
    if (z != 0)
    {
      c += "  if (Z + " + z_s + " >= args.dst_tensor.Slices()) return;\n";
    }
    for (int y = 0; y < block_size.y; ++y)
    {
      const std::string y_s = std::to_string(y);
      for (int x = 0; x < block_size.x * element_size; ++x)
      {
        const std::string x_s = std::to_string(x);
        c += "  if (X + " + x_s + " < args.dst_tensor.Width() && Y + " + y_s +
             " < args.dst_tensor.Height()) {\n";
        c += "    FLT4 res = TO_FLT4(r" + z_s + y_s + x_s + ");\n";
        c += "    args.dst_tensor.Write(res, X + " + x_s + ", Y + " + y_s + ", Z + " + z_s + ");\n";
        c += "  }\n";
      }
    }
  }
  c += "}\n";
  return c;
}

int3 ConvBuffer1x1::GetGridSize() const
{
  const int dst_width_elements =
    DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(), (conv_params_.element_size / 4));
  const int grid_x = DivideRoundUp(dst_width_elements, conv_params_.block_size.x);
  const int grid_y = DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int grid_z = DivideRoundUp(dst_[0]->Slices(), conv_params_.block_size.z);
  return int3(grid_x, grid_y, grid_z);
}

void ConvBuffer1x1::GetPossibleKernelWorkGroups(TuningType tuning_type,
                                                const DeviceInfo &device_info,
                                                const KernelInfo &kernel_info,
                                                std::vector<int3> *work_groups) const
{
  GetPossibleWorkGroupsConv(tuning_type, device_info, kernel_info, grid_size_, work_groups);
}

bool IsConvBuffer1x1Supported(const OperationDef &definition, const Convolution2DAttributes &attr)
{
  auto src_storage_type = definition.src_tensors[0].storage_type;
  return src_storage_type == TensorStorageType::BUFFER && attr.weights.shape.w == 1 &&
         attr.weights.shape.h == 1 && attr.dilations.w == 1 && attr.dilations.h == 1 &&
         attr.strides.w == 1 && attr.strides.h == 1 && attr.padding.prepended.w == 0 &&
         attr.padding.prepended.h == 0 && attr.padding.appended.w == 0 &&
         attr.padding.appended.h == 0;
}

bool IsConvBuffer1x1Supported(const OperationDef &definition, const BHWC &weights_shape,
                              const Convolution2DAttributes &attr)
{
  auto src_storage_type = definition.src_tensors[0].storage_type;
  return src_storage_type == TensorStorageType::BUFFER && weights_shape.w == 1 &&
         weights_shape.h == 1 && attr.dilations.w == 1 && attr.dilations.h == 1 &&
         attr.strides.w == 1 && attr.strides.h == 1 && attr.padding.prepended.w == 0 &&
         attr.padding.prepended.h == 0 && attr.padding.appended.w == 0 &&
         attr.padding.appended.h == 0;
}

ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info, const OperationDef &definition,
                                  const Convolution2DAttributes &attr, const BHWC *shape)
{
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape)
  {
    conv_params = GetBestParams(device_info, definition, *shape, src_depth, dst_depth);
  }
  else
  {
    conv_params = GetBestParams(device_info, definition, src_depth, dst_depth);
  }
  ConvBuffer1x1 result(definition, conv_params);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info, const OperationDef &definition,
                                  const FullyConnectedAttributes &attr, const BHWC *shape)
{
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape)
  {
    conv_params = GetBestParams(device_info, definition, *shape, src_depth, dst_depth);
  }
  else
  {
    conv_params = GetBestParams(device_info, definition, src_depth, dst_depth);
  }
  conv_params.block_size.x *= conv_params.block_size.y;
  conv_params.block_size.y = 1;
  ConvBuffer1x1 result(definition, conv_params);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(const DeviceInfo &device_info,
                                              const OperationDef &definition,
                                              const Convolution2DAttributes &attr,
                                              const BHWC *shape)
{
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape)
  {
    conv_params = GetBestParams(device_info, definition, *shape, src_depth, dst_depth);
  }
  else
  {
    conv_params = GetBestParams(device_info, definition, src_depth, dst_depth);
  }
  conv_params.block_size.x *= conv_params.block_size.y;
  conv_params.block_size.y = 1;
  conv_params.different_weights_for_height = true;
  ConvBuffer1x1 result(definition, conv_params);
  result.UploadDataForWinograd4x4To6x6(attr.weights);
  return result;
}

ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(const DeviceInfo &device_info,
                                                const OperationDef &definition,
                                                const Convolution2DAttributes &attr,
                                                const BHWC &weights_shape, const BHWC *dst_shape)
{
  const int dst_depth = DivideRoundUp(weights_shape.b, 4);
  const int src_depth = DivideRoundUp(weights_shape.c, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (dst_shape)
  {
    conv_params = GetBestParams(device_info, definition, *dst_shape, src_depth, dst_depth);
  }
  else
  {
    conv_params = GetBestParams(device_info, definition, src_depth, dst_depth);
  }
  ConvBuffer1x1 result(definition, conv_params);
  result.UploadBiases(attr.bias);
  return result;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
