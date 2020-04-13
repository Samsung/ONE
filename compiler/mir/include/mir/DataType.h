/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_DATA_TYPE_H_
#define _MIR_DATA_TYPE_H_

#include <cassert>
#include <cstdint>

namespace mir
{

enum class DataType
{
  UNKNOWN,
  FLOAT32,
  FLOAT64,
  INT32,
  INT64,
  UINT8
};

inline std::size_t getDataTypeSize(DataType type)
{
  switch (type)
  {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT64:
      return sizeof(double);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return 0;
  }
}

} // namespace mir

#endif //_MIR_DATA_TYPE_H_
