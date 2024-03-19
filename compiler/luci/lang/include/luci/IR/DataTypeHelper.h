/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_LANG_IR_DATA_TYPE_HELPER_H__
#define __LUCI_LANG_IR_DATA_TYPE_HELPER_H__

#include <loco/IR/DataType.h>
#include <loco/IR/DataTypeTraits.h>

namespace luci
{

/**
 * @brief Returns the size of the data type.
 * @note  luci saves S4, U4 in a single byte.
 *        The extra 4 bits in the MSB side are filled with sign bits.
 */
inline uint32_t size(loco::DataType data_type)
{
  switch (data_type)
  {
    case loco::DataType::S4:
      return sizeof(loco::DataTypeImpl<loco::DataType::S4>::Type);
    case loco::DataType::U4:
      return sizeof(loco::DataTypeImpl<loco::DataType::U4>::Type);
    default:
      return loco::size(data_type);
  }
}

} // namespace luci

#endif // __LUCI_LANG_IR_DATA_TYPE_HELPER_H__
