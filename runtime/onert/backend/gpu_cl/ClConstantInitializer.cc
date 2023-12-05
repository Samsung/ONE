/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ClConstantInitializer.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

ClConstantInitializer::ClConstantInitializer(const ir::Operands &operands,
                                             const std::shared_ptr<ITensorRegistry> &tensor_reg)
  : _operands{operands}, _tensor_reg{tensor_reg}
{
  // DO NOTHING
}

void ClConstantInitializer::copyInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  if (input_index.valid())
  {
    const auto &input_obj = _operands.at(input_index);
    registerCopyInitializer(input_index, input_obj);
  }
}

void ClConstantInitializer::permuteInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  const auto &input_obj = _operands.at(input_index);
  registerPermuteInitializer(input_index, input_obj);
}

// NOTE Workaround for 16b float type. Here, this is enough since only the size of bytes matters.
using float16 = uint16_t;

void ClConstantInitializer::registerCopyInitializer(const ir::OperandIndex &index,
                                                    const ir::Operand &obj)
{
  // For only CONSTANTS
  // TODO Add to check if tensor has been allocated
  if (!obj.isConstant())
    return;

  const auto type = obj.typeInfo().type();
  using ir::DataType;

  switch (type)
  {
    case DataType::FLOAT32:
      _init_map[index] = copyInit<float>;
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

void ClConstantInitializer::registerPermuteInitializer(const ir::OperandIndex &index,
                                                       const ir::Operand &obj)
{
  // For only CONSTANTS
  // TODO Add to check if tensor has been allocated
  if (!obj.isConstant())
    return;

  const auto type = obj.typeInfo().type();
  const auto frontend_layout = obj.info().layout();
  using ir::DataType;
  using namespace std::placeholders;

  switch (type)
  {
    case DataType::FLOAT32:
      _init_map[index] = std::bind(permuteInit<float>, _1, _2, frontend_layout);
      break;
    case DataType::INT32:
      _init_map[index] = std::bind(permuteInit<int32_t>, _1, _2, frontend_layout);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
