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

#include "Transpose.h"

#include <string>

#include "Util.h"
#include "WorkGroupPicking.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace
{
std::string GetTransposeCode(const OperationDef &op_def, const TransposeAttributes &attr)
{
  const std::string batch_id = op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
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
  c += "  FLT temps[4];\n";
  c += "  temps[0] = (FLT)(0.0f);\n";
  c += "  temps[1] = (FLT)(0.0f);\n";
  c += "  temps[2] = (FLT)(0.0f);\n";
  c += "  temps[3] = (FLT)(0.0f);\n";
  int remap[4];
  remap[attr.perm.b] = 0;
  remap[attr.perm.h] = 1;
  remap[attr.perm.w] = 2;
  remap[attr.perm.c] = 3;
  if (attr.perm.c == 3)
  { // optimized reading when no channels permutation
    const std::string bhw[] = {batch_id, "Y", "X"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH))
    {
      c += "  args.src_tensor.SetBatchRef(" + bhw[remap[0]] + ");\n";
    }
    c += "  int s_y = " + bhw[remap[1]] + ";\n";
    c += "  int s_x = " + bhw[remap[2]] + ";\n";
    c += "  FLT4 t = args.src_tensor.Read(s_x, s_y, Z);\n";
    c += "  temps[0] = t.x;\n";
    c += "  temps[1] = t.y;\n";
    c += "  temps[2] = t.z;\n";
    c += "  temps[3] = t.w;\n";
  }
  else
  {
    c += "  for (int i = 0; i < 4; ++i) {\n";
    c += "    int dst_channel = Z * 4 + i;\n";
    c += "    if (dst_channel < args.dst_tensor.Channels()) {\n";
    const std::string bhwc[] = {batch_id, "Y", "X", "dst_channel"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH))
    {
      c += "      args.src_tensor.SetBatchRef(" + bhwc[remap[0]] + ");\n";
    }
    c += "      int s_y = " + bhwc[remap[1]] + ";\n";
    c += "      int s_x = " + bhwc[remap[2]] + ";\n";
    c += "      int s_c = " + bhwc[remap[3]] + ";\n";
    c += "      int s_z = s_c / 4;\n";
    c += "      int src_sub_ch = s_c % 4;\n";
    c += "      FLT4 t = args.src_tensor.Read(s_x, s_y, s_z);\n";
    c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
    c += "      temps[i] = t_ar[src_sub_ch];\n";
    c += "    }\n";
    c += "  }\n";
  }
  c += "  FLT4 result = (FLT4)(temps[0], temps[1], temps[2], temps[3]);\n";
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}
} // namespace

GPUOperation CreateTranspose(const OperationDef &definition, const TransposeAttributes &attr)
{
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetTransposeCode(definition, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
