/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backend/IConstantInitializer.h"

#include <Half.h>

using float16 = Half;

namespace onert
{
namespace backend
{

void IConstantInitializer::registerCopyInitializer(const ir::OperandIndex &index,
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
    case DataType::INT32:
      _init_map[index] = copyInit<int32_t>;
      break;
    case DataType::UINT32:
      _init_map[index] = copyInit<uint32_t>;
      break;
    case DataType::BOOL8:
    case DataType::QUANT_UINT8_ASYMM:
      _init_map[index] = copyInit<uint8_t>;
      break;
    case DataType::QUANT_INT8_SYMM:
    case DataType::QUANT_INT8_ASYMM:
      _init_map[index] = copyInit<int8_t>;
      break;
    case DataType::FLOAT16:
      _init_map[index] = copyInit<float16>;
      break;
    case DataType::INT64:
      _init_map[index] = copyInit<int64_t>;
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

void IConstantInitializer::registerPermuteInitializer(const ir::OperandIndex &index,
                                                      const ir::Operand &obj)
{
  // For only CONSTANTS
  // TODO Add to check if tensor has been allocated
  if (!obj.isConstant())
    return;

  const auto type = obj.typeInfo().type();
  using ir::DataType;
  using namespace std::placeholders;

  switch (type)
  {
    case DataType::FLOAT32:
      _init_map[index] = std::bind(permuteInit<float>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::INT32:
      _init_map[index] = std::bind(permuteInit<int32_t>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::UINT32:
      _init_map[index] = std::bind(permuteInit<uint32_t>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::BOOL8:
    case DataType::QUANT_UINT8_ASYMM:
      _init_map[index] = std::bind(permuteInit<uint8_t>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::QUANT_INT8_SYMM:
    case DataType::QUANT_INT8_ASYMM:
      _init_map[index] = std::bind(permuteInit<int8_t>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::FLOAT16:
      _init_map[index] = std::bind(permuteInit<float16>, _1, _2, _current_op_seq_layout);
      break;
    case DataType::INT64:
      _init_map[index] = std::bind(permuteInit<int64_t>, _1, _2, _current_op_seq_layout);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

} // namespace backend
} // namespace onert
