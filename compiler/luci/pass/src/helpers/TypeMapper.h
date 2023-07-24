/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PASS_HELPERS_TYPE_MAPPER_H__
#define __LUCI_PASS_HELPERS_TYPE_MAPPER_H__

#include <loco/IR/DataType.h>

#include <cstdint>

namespace luci
{

/**
 * @brief TypeMapper maps between c++ primitive data type and loco::DataType.
 */
template <typename T> struct TypeMapper
{
  static constexpr loco::DataType get() { return loco::DataType::Unknown; }
};

template <> struct TypeMapper<float>
{
  static constexpr loco::DataType get() { return loco::DataType::FLOAT32; }
};

template <> struct TypeMapper<uint8_t>
{
  static constexpr loco::DataType get() { return loco::DataType::U8; }
};

template <> struct TypeMapper<uint16_t>
{
  static constexpr loco::DataType get() { return loco::DataType::U16; }
};

template <> struct TypeMapper<uint32_t>
{
  static constexpr loco::DataType get() { return loco::DataType::U32; }
};

template <> struct TypeMapper<uint64_t>
{
  static constexpr loco::DataType get() { return loco::DataType::U64; }
};

template <> struct TypeMapper<int8_t>
{
  static constexpr loco::DataType get() { return loco::DataType::S8; }
};

template <> struct TypeMapper<int16_t>
{
  static constexpr loco::DataType get() { return loco::DataType::S16; }
};

template <> struct TypeMapper<int32_t>
{
  static constexpr loco::DataType get() { return loco::DataType::S32; }
};

template <> struct TypeMapper<int64_t>
{
  static constexpr loco::DataType get() { return loco::DataType::S64; }
};

} // namespace luci

#endif // __LUCI_PASS_HELPERS_TYPE_MAPPER_H__
