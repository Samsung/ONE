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

#ifndef __LUCI_IR_CIRCLE_REMOTE_CONST_H__
#define __LUCI_IR_CIRCLE_REMOTE_CONST_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

#include <loco/IR/DataTypeTraits.h>

namespace luci
{

/**
 * @brief Class for reference to tensor data
 * @note  This will not be exported as a specific op. CircleRemoteConst has access to provided data
 * and provides reading access to user.
 */
class CircleRemoteConst final : public FixedArityNode<0, CircleNodeImpl<CircleOpcode::CIRCLECONST>>
{
public:
  template <loco::DataType DT> uint32_t size(void) const;

  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &scalar(void) const;

  // Note: this function makes reference to remote data buffer, CircleRemoteConst not owns this data
  void bind_buffer(const uint8_t *data, uint32_t size);
  const uint8_t *data() const;
  uint32_t buffer_size() const;

private:
  struct RemoteBuffer
  {
    const uint8_t *data = nullptr;
    uint32_t size = 0;
  };

private:
  RemoteBuffer _buffer;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_REMOTE_CONST_H__
