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

#include "Reshape.h"

#include <string>

#include "Util.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{

std::string GetReshapeCode(const OperationDef &op_def)
{
  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH))
  {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  else
  {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH))
  {
    c += "  int dst_bhwc4 = B;\n";
  }
  else
  {
    c += "  int dst_bhwc4 = 0;\n";
  }
  c += "  dst_bhwc4 = ((dst_bhwc4 * args.dst_tensor.Height() + Y) * "
       "args.dst_tensor.Width() + X) * args.dst_tensor.Slices() + Z;\n";
  c += "  int src_z = dst_bhwc4 % args.src_tensor.Slices();\n";
  c += "  dst_bhwc4 = dst_bhwc4 / args.src_tensor.Slices();\n";
  c += "  int src_x = dst_bhwc4 % args.src_tensor.Width();\n";
  c += "  dst_bhwc4 = dst_bhwc4 / args.src_tensor.Width();\n";
  c += "  int src_y = dst_bhwc4 % args.src_tensor.Height();\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH))
  {
    c += "  int src_b = dst_bhwc4 / args.src_tensor.Height();\n";
    c += "  args.src_tensor.SetBatchRef(src_b);\n";
  }
  c += "  FLT4 result = args.src_tensor.Read(src_x, src_y, src_z);\n";
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}

} // namespace

GPUOperation CreateReshapex4(const OperationDef &definition)
{
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetReshapeCode(definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
