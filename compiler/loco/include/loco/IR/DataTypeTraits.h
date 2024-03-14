/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_DATA_TYPE_TRAITS_H__
#define __LOCO_IR_DATA_TYPE_TRAITS_H__

#include "loco/IR/DataType.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace loco
{

/**
 * @brief C++ scalar type corresponding to each DataType
 */
template <DataType DT> struct DataTypeImpl
{
  // using Type = ...
};

// TODO Support other enum values
template <> struct DataTypeImpl<DataType::S4>
{
  // Use C++ int8_t type for 4bit integer
  using Type = int8_t;
};

template <> struct DataTypeImpl<DataType::S8>
{
  // Use C++ int8_t type for 8bit integer
  using Type = int8_t;
};

template <> struct DataTypeImpl<DataType::U4>
{
  // Use C++ uint8_t type for unsigned 4bit integer
  using Type = uint8_t;
};

template <> struct DataTypeImpl<DataType::U8>
{
  // Use C++ uint8_t type for unsigned 8bit integer
  using Type = uint8_t;
};

template <> struct DataTypeImpl<DataType::S16>
{
  // Use C++ int16_t type for 16bit integer
  using Type = int16_t;
};

template <> struct DataTypeImpl<DataType::U16>
{
  // Use C++ uint16_t type for unsigned 16bit integer
  using Type = uint16_t;
};

template <> struct DataTypeImpl<DataType::S32>
{
  // Use C++ int32_t type for 32bit integer
  using Type = int32_t;
};

template <> struct DataTypeImpl<DataType::U32>
{
  // Use C++ uint32_t type for unsigned 32bit integer
  using Type = uint32_t;
};

template <> struct DataTypeImpl<DataType::S64>
{
  // Use C++ int64_t type for 64bit integer
  using Type = int64_t;
};

template <> struct DataTypeImpl<DataType::U64>
{
  // Use C++ uint64_t type for unsigned 64bit integer
  using Type = uint64_t;
};

template <> struct DataTypeImpl<DataType::FLOAT16>
{
  // float16 type with 16bit value, encoded with help of FP16 library
  // https://github.com/Maratyszcza/FP16/
  using Type = uint16_t;
};

template <> struct DataTypeImpl<DataType::FLOAT32>
{
  // Use C++ float type for IEEE 32-bit floating-point numbers
  using Type = float;
};

template <> struct DataTypeImpl<DataType::FLOAT64>
{
  // Use C++ double type for IEEE 64-bit floating-point numbers
  using Type = double;
};

// NOTE DataTypeImpl for BOOL is subject to change
template <> struct DataTypeImpl<DataType::BOOL>
{
  // Use C++ uint8_t type for bool
  using Type = uint8_t;
};

template <> struct DataTypeImpl<DataType::STRING>
{
  // Use C++ std::string type for STRING
  using Type = std::string;
};

/**
 * @brief Returns the size of the data type.
 * @note If you need the size at compile time, use `sizeof(typename DataTypeImpl<DT>::Type)`.
 */
inline uint32_t size(DataType data_type)
{
  switch (data_type)
  {
    case DataType::S4:
      throw std::runtime_error("S4 type is not supported by loco::size");
    case DataType::S8:
      return sizeof(DataTypeImpl<DataType::S8>::Type);
    case DataType::U4:
      throw std::runtime_error("U4 type is not supported by loco::size");
    case DataType::U8:
      return sizeof(DataTypeImpl<DataType::U8>::Type);
    case DataType::S16:
      return sizeof(DataTypeImpl<DataType::S16>::Type);
    case DataType::U16:
      return sizeof(DataTypeImpl<DataType::U16>::Type);
    case DataType::S32:
      return sizeof(DataTypeImpl<DataType::S32>::Type);
    case DataType::U32:
      return sizeof(DataTypeImpl<DataType::U32>::Type);
    case DataType::S64:
      return sizeof(DataTypeImpl<DataType::S64>::Type);
    case DataType::U64:
      return sizeof(DataTypeImpl<DataType::U64>::Type);
    case DataType::FLOAT16:
      return sizeof(DataTypeImpl<DataType::FLOAT16>::Type);
    case DataType::FLOAT32:
      return sizeof(DataTypeImpl<DataType::FLOAT32>::Type);
    case DataType::FLOAT64:
      return sizeof(DataTypeImpl<DataType::FLOAT64>::Type);
    case DataType::BOOL:
      return sizeof(DataTypeImpl<DataType::BOOL>::Type);
    case DataType::STRING:
      // STRING is variable length. Cannot decide size by type
      throw std::runtime_error("Invalid size call with STRING type");
    default:
      // TODO Support remaining data types.
      assert(false);
      return UINT32_MAX; // Avoid compiler warning.
  }
}

} // namespace loco

#endif // __LOCO_IR_DATA_TYPE_TRAITS_H__
