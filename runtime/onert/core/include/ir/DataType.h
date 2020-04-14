/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_DATATYPE_H__
#define __ONERT_IR_DATATYPE_H__

#include <stdexcept>

namespace onert
{
namespace ir
{

enum class DataType
{
  FLOAT32 = 0,
  INT32 = 1,
  UINT32 = 2,
  QUANT8_ASYMM = 3,
  BOOL8 = 4,
  UINT8 = 5,
  QUANT8_SYMM = 6,
};

inline size_t sizeOfDataType(DataType data_type)
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
    case DataType::QUANT8_ASYMM:
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::QUANT8_SYMM:
      return sizeof(int8_t);
    default:
      throw std::runtime_error{"Unsupported type size"};
  }
}

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_DATATYPE_H__
