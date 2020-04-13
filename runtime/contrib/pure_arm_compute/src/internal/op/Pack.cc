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

#include "internal/op/Pack.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Pack
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace Pack
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace Pack
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(outputCount == 1);

  // Each input should be interpreted as follows:
  //
  //  0 .. n - 3 -> Input Tensor Index
  //  n - 2      -> Input Tensor counts (will be ignored)
  //  n - 1      -> Input Axis Index
  ofm_index = outputs[0];
  axis_index = inputs[inputCount - 1];
  // last input is axis along which packing is required
  for (uint32_t n = 0; n < inputCount - 2; ++n)
  {
    ifm_indexes.emplace_back(inputs[n]);
  }
}

} // namespace Pack
} // namespace op
} // namespace tflite
} // namespace internal
