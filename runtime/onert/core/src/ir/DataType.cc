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

#include "ir/DataType.h"

#include <stdexcept>
#include <Half.h>

using float16 = Half;

namespace onert
{
namespace ir
{

size_t sizeOfDataType(DataType data_type)
{
  switch (data_type)
  {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::UINT32:
      return sizeof(uint32_t);
    case DataType::BOOL8:
    case DataType::QUANT_UINT8_ASYMM:
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::QUANT_INT8_SYMM:
    case DataType::QUANT_INT8_ASYMM:
    case DataType::QUANT_INT8_SYMM_PER_CHANNEL:
      return sizeof(int8_t);
    case DataType::FLOAT16:
      return sizeof(float16);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::QUANT_INT16_ASYMM:
      return sizeof(int16_t);
    case DataType::QUANT_INT16_SYMM:
      return sizeof(int16_t);
    case DataType::QUANT_INT4_SYMM:
      return sizeof(uint8_t); // Q: what is type size for int4?
    case DataType::QUANT_UINT4_ASYMM:
      return sizeof(uint8_t); // Q: what is type size for uint4?
    default:
      throw std::runtime_error{"Unsupported type size"};
  }
}

} // namespace ir
} // namespace onert
