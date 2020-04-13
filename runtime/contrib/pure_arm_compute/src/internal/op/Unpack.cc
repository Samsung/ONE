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

#include "internal/op/Unpack.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Unpack
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace Unpack
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace Unpack
{
// There are three inputs: tensor which is to be unpacked,
// axis along which tensor needs to be unpacked
// and number of splits along the axis.

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 3);

  ifm_index = inputs[0];

  for (uint32_t n = 0; n < outputCount; ++n)
  {
    ofm_indexes.emplace_back(outputs[n]);
  }
  num_split_index = inputs[1];
  axis_index = inputs[2];
}

} // namespace Unpack
} // namespace op
} // namespace tflite
} // namespace internal
