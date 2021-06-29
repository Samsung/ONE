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

#include "Elementwise.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "absl/strings/substitute.h"
#include "GpuOperation.h"
#include "open_cl/Operations.h"

#include "Util.h"
#include "open_cl/StorageTypeUtil.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

namespace
{

std::string GetTwoInputCode(const OperationType &op_type, const std::string &result_var,
                            const std::string &input0, const std::string &input1,
                            bool swap_inputs = false)
{
  std::string result;
  switch (op_type)
  {
    case OperationType::ADD:
      result += "$0 = $1 + $2;\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  if (swap_inputs)
  {
    return absl::Substitute(result, result_var, input1, input0);
  }
  else
  {
    return absl::Substitute(result, result_var, input0, input1);
  }
}

} // namespace

GPUOperation CreateElementwiseTwoInput(const OperationDef &definition, const OperationType &op_type,
                                       const BHWC &shape)
{
  GPUOperation op(definition);
  op.elementwise_ = true;
  auto src_desc = definition.src_tensors[1];
  if (definition.IsBatchSupported())
  {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddSrcTensor("second_tensor", src_desc);
  const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  op.code_ = absl::StrCat("FLT4 second_val = args.second_tensor.Read(", x_coord, ", ", y_coord,
                          ", ", s_coord, ");\n");
  if (shape.c == 1)
  {
    op.code_ += "  second_val.y = second_val.x;\n";
    op.code_ += "  second_val.z = second_val.x;\n";
    op.code_ += "  second_val.w = second_val.x;\n";
  }
  op.code_ += GetTwoInputCode(op_type, "in_out_value", "in_out_value", "second_val", false);
  return op;
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
