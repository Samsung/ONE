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

#ifndef __LUCI_IR_CIRCLECONST_H__
#define __LUCI_IR_CIRCLECONST_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

#include <loco/IR/DataTypeTraits.h>

namespace luci
{

/**
 * @brief Class to build tensor data
 * @note  This will not be exported as a specific op
 */
class CircleConst final : public FixedArityNode<0, CircleNodeImpl<CircleOpcode::CIRCLECONST>>
{
public:
  CircleConst() = default;

public:
  uint32_t size(void) const;
  void size(uint32_t size);

  template <loco::DataType DT> uint32_t size(void) const;
  template <loco::DataType DT> void size(uint32_t size);
  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
  template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &at(uint32_t n);

  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &scalar(void) const;
  template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &scalar(void);

private:
  std::vector<uint8_t> _data;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLECONST_H__
