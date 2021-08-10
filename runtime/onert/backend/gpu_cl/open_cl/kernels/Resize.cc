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

#include "Resize.h"

#include <string>

#include "Util.h"
#include "GpuOperation.h"
#include "open_cl/Operations.h"
#include "open_cl/StorageTypeUtil.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

Resize::Resize(const OperationDef &definition, const Resize2DAttributes &attr)
  : GPUOperation(definition), attr_(attr)
{
  code_ = GetResizeCode(definition_, attr_);
}

Resize::Resize(Resize &&operation) : GPUOperation(std::move(operation)), attr_(operation.attr_) {}

Resize &Resize::operator=(Resize &&operation)
{
  if (this != &operation)
  {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize::GetResizeCode(const OperationDef &op_def, const Resize2DAttributes &attr)
{
  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported())
  {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported())
  {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);
  args_.AddInt("border_x");
  args_.AddInt("border_y");
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  if (op_def.IsBatchSupported())
  {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  if (linear_id >= args.dst_tensor.Width() || Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  }
  else
  {
    c += "  int X = get_global_id(0);\n";
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| Z >= args.dst_tensor.Slices()) return;\n";
  }
  if (attr.type == SamplingType::NEAREST)
  {
    std::string fxc;
    std::string fyc;
    if (attr.half_pixel_centers)
    {
      fxc = "(X + 0.5f) * args.scale_factor_x";
      fyc = "(Y + 0.5f) * args.scale_factor_y";
    }
    else
    {
      fxc = "X * args.scale_factor_x";
      fyc = "Y * args.scale_factor_y";
    }
    if (attr.align_corners)
    {
      fxc += " + 0.5f";
      fyc += " + 0.5f";
    }
    c += "  int2 coord;\n";
    c += "  coord.x = (int)(" + fxc + ");\n";
    c += "  coord.y = (int)(" + fyc + ");\n";
    c += "  coord.x = max(0, coord.x);\n";
    c += "  coord.y = max(0, coord.y);\n";
    c += "  coord.x = min(coord.x, args.border_x);\n";
    c += "  coord.y = min(coord.y, args.border_y);\n";
    if (op_def.IsBatchSupported())
    {
      c += "  coord.x = coord.x * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  FLT4 r0 = args.src_tensor.Read(coord.x, coord.y, Z);\n";
  }
  else
  {
    if (attr.half_pixel_centers)
    {
      c += "  float2 f_coords = ((float2)(X, Y) + 0.5f) * "
           "(float2)(args.scale_factor_x, args.scale_factor_y) - "
           "0.5f;\n";
    }
    else
    {
      c += "  float2 f_coords = (float2)(X, Y) * (float2)(args.scale_factor_x, "
           "args.scale_factor_y);\n";
    }
    c += "  float2 f_coords_floor = floor(f_coords);\n";
    c += "  int2 coords_floor = (int2)(f_coords_floor.x, f_coords_floor.y);\n";
    c += "  int4 st;\n";
    c += "  st.xy = max(coords_floor, (int2)(0, 0));\n";
    c += "  st.zw = min(coords_floor + (int2)(1, 1), (int2)(args.border_x, "
         "args.border_y));\n";
    c += "  float2 t = f_coords - f_coords_floor;\n";
    if (op_def.IsBatchSupported())
    {
      c += "  st.x = st.x * args.src_tensor.Batch() + B;\n";
      c += "  st.z = st.z * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  float4 src0 = args.src_tensor.Read<float>(st.x, st.y, Z);\n";
    c += "  float4 src1 = args.src_tensor.Read<float>(st.z, st.y, Z);\n";
    c += "  float4 src2 = args.src_tensor.Read<float>(st.x, st.w, Z);\n";
    c += "  float4 src3 = args.src_tensor.Read<float>(st.z, st.w, Z);\n";
    c += "  FLT4 r0 = TO_FLT4(mix(mix(src0, src1, t.x), mix(src2, src3, t.x), "
         "t.y));\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z);\n";
  c += "}\n";
  return c;
}

absl::Status Resize::BindArguments(ArgumentsBinder *args)
{
  RETURN_IF_ERROR(args->SetInt("border_x", src_[0]->Width() - 1));
  RETURN_IF_ERROR(args->SetInt("border_y", src_[0]->Height() - 1));
  RETURN_IF_ERROR(args->SetFloat("scale_factor_x",
                                 CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args->SetFloat(
    "scale_factor_y", CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  return absl::OkStatus();
}

int3 Resize::GetGridSize() const
{
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Resize CreateResize(const OperationDef &definition, const Resize2DAttributes &attr)
{
  return Resize(definition, attr);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
