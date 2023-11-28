/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ValidateHelpers.h"

namespace luci
{

bool validate_batch_space_nd(const GraphBuilderBase::ValidateArgs &args)
{
  const auto &inputs = args.op.inputs;
  if (inputs.size() != 3)
    return false;

  // input 1 and 2 should have INT32/INT64 type
  const auto tensors = args.reader.tensors();
  const auto tensor_1 = tensors.at(inputs.at(1));
  assert(tensor_1 != nullptr);
  switch (tensor_1->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }
  const auto tensor_2 = tensors.at(inputs.at(2));
  assert(tensor_2 != nullptr);
  switch (tensor_2->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  // Only support input shape dimension 3 and 4 only
  const auto tensor_0 = tensors.at(inputs.at(0));
  assert(tensor_0 != nullptr);
  const auto t_0_s = wrap(tensor_0->shape()).size();
  if (t_0_s != 3 && t_0_s != 4)
    return false;

  // TODO check input shape

  return true;
}

bool validate_minmax(const GraphBuilderBase::ValidateArgs &args)
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;

  if (outputs.size() != 1)
    return false;

  const auto tensors = args.reader.tensors();
  const auto tensor = tensors.at(inputs.at(0));
  assert(tensor != nullptr);
  switch (tensor->type())
  {
    case circle::TensorType_FLOAT16:
    case circle::TensorType_FLOAT32:
    case circle::TensorType_FLOAT64:
    case circle::TensorType_INT16:
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
    case circle::TensorType_UINT8:
      break;
    default:
      return false;
  }

  assert(tensors[inputs.at(1)] != nullptr);
  if (tensors[inputs.at(1)]->type() != tensor->type())
    return false;

  assert(tensors[outputs[0]] != nullptr);
  if (tensors[outputs[0]]->type() != tensor->type())
    return false;

  return true;
}

bool validate_reduce_minmax(const GraphBuilderBase::ValidateArgs &args)
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;

  if (outputs.size() != 1)
    return false;

  const auto tensors = args.reader.tensors();
  const auto tensor_axis = tensors.at(inputs.at(1));
  assert(tensor_axis != nullptr);
  switch (tensor_axis->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  return true;
}

} // namespace luci
