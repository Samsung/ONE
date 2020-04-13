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

#ifndef __INTERNAL_OP_UNPACK_H__
#define __INTERNAL_OP_UNPACK_H__

#include "internal/op/Node.h"

#include <cstdint>
#include <vector>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Unpack
{

struct Param
{
  int32_t ifm_index;
  int32_t axis_index;
  int32_t num_split_index;
  // There are N outputs after Unpacking Input Tensor along axis
  std::vector<int32_t> ofm_indexes;

  Param() = default;
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

class Node final : public op::Node
{
public:
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  virtual ~Node() = default;

public:
  const Param &param(void) const { return _param; }

public:
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace Unpack
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_UNPACK_H__
