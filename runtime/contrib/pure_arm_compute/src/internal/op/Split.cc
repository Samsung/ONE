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

#include "internal/op/Split.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Split
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace Split
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace Split
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 3);

  // Each input should be interpreted as follows:
  //  0 -> An n-D tensor, specifying the tensor to be split.
  //  1 -> A 0-D int32 tensor, indicating the dimension along which to split.
  //  2 -> A 0-D int32 tensor, indicating the number of outputs
  //       (It can be ignored on pacl becasue pacl don't support dynamic tensor shape,
  //        and can be used for verification only)
  ifm_index = inputs[0];
  axis_index = inputs[1];

  // Each output should be interpreted as follow:
  //  [0, outputCount) -> An n-D tensor.
  for (uint32_t n = 0; n < outputCount; ++n)
  {
    ofm_indexes.emplace_back(outputs[n]);
  }
}

} // namespace Split
} // namespace op
} // namespace tflite
} // namespace internal
