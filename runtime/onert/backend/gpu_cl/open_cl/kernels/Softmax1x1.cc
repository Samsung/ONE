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

#include "Softmax1x1.h"

#include <string>

#include "Util.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

Softmax1x1::Softmax1x1(const OperationDef &definition) : GPUOperation(definition)
{
  work_group_size_ = int3(32, 1, 1);
  code_ = GetSoftmaxKernelCode(definition_);
}

Softmax1x1::Softmax1x1(Softmax1x1 &&kernel) : GPUOperation(std::move(kernel)) {}

Softmax1x1 &Softmax1x1::operator=(Softmax1x1 &&kernel)
{
  if (this != &kernel)
  {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string Softmax1x1::GetSoftmaxKernelCode(const OperationDef &op_def)
{
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");
  args_.AddInt("slices_x32");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  if (op_def.IsBatchSupported())
  {
    c += "  int batch_id = get_global_id(1);\n";
    c += "  if (batch_id >= args.dst_tensor.Batch()) return;\n";
    c += "  args.dst_tensor.SetBatchRef(batch_id);\n";
    c += "  args.src_tensor.SetBatchRef(batch_id);\n";
  }
  c += "  float4 mask = (float4)(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c += "  int offset = 0;\n";
  c += "  float sum = 0.0f;\n";
  c += "  int s = 0;\n";
  c += "  int tid = get_local_id(0);\n";
  c += "  do {\n";
  c += "    int z = offset + tid;\n";
  c += "    if (z < args.dst_tensor.Slices()) {\n";
  c += "      float4 mask_temp = z == args.dst_tensor.Slices() - 1 ? mask : "
       "(float4)(1.0f);\n";
  c += "      float4 src = args.src_tensor.Read<float>(0, 0, z);\n";
  c += "      sum += dot(mask_temp, exp(src));\n";
  c += "      offset += 32;\n";
  c += "    }\n";
  c += "    s++;\n";
  c += "  } while (s < args.slices_x32);\n";
  c += "\n";
  c += "  __local float4 tmp[8];\n";
  c += "  __local float* tmpx1 = (__local float*)tmp;\n";
  c += "  tmpx1[tid] = sum;\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  c += "  if (tid == 0) {\n";
  c += "    sum = dot((float4)(1.0f), tmp[0]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[1]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[2]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[3]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[4]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[5]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[6]);\n";
  c += "    sum += dot((float4)(1.0f), tmp[7]);\n";
  c += "    tmpx1[0] = 1.0f / sum;\n";
  c += "  }\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  c += "  sum = tmpx1[0];\n";
  c += "\n";
  c += "  offset = 0;\n";
  c += "  s = 0;\n";
  c += "  do {\n";
  c += "    int z = offset + tid;\n";
  c += "    if (z < args.dst_tensor.Slices()) {\n";
  c += "      FLT4 res = TO_FLT4(exp(args.src_tensor.Read<float>(0, 0, "
       "z))*sum);\n";
  c += "      args.dst_tensor.Write(res, 0, 0, z);\n";
  c += "      offset += 32;\n";
  c += "    }\n";
  c += "    s++;\n";
  c += "  } while (s < args.slices_x32);\n";
  c += "}\n";
  return c;
}

absl::Status Softmax1x1::BindArguments(ArgumentsBinder *args)
{
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
  RETURN_IF_ERROR(args->SetInt("slices_x32", DivideRoundUp(src_[0]->Slices(), 32)));
  return absl::OkStatus();
}

int3 Softmax1x1::GetGridSize() const { return int3(32, dst_[0]->Batch(), 1); }

Softmax1x1 CreateSoftmax1x1(const OperationDef &definition) { return Softmax1x1(definition); }

} // namespace gpu_cl
} // namespace backend
} // namespace onert
