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

#include "internal/op/MaxPool2D.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace MaxPool2D
{
namespace Explicit
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace Explicit

namespace Implicit
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace Implicit
} // namespace MaxPool2D
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace MaxPool2D
{
namespace Explicit
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 10 && outputCount == 1);

  ofm_index = outputs[0];

  // Each input should be interpreted as follows:
  //
  //  0 -> IFM Tensor Index
  //  1 -> Padding_left index
  //  2 -> Padding_right index
  //  3 -> Padding_top index
  //  4 -> Padding_bottom index
  //  5 -> Horizontal (over width) Stride Index
  //  6 -> Vertial (over height) Stride Index
  //  7 -> Filter Width Index
  //  8 -> Filter Height Index
  //  9 -> FuseCode (activation) Index
  ifm_index = inputs[0];
  padding_left_index = inputs[1];
  padding_right_index = inputs[2];
  padding_top_index = inputs[3];
  padding_bottom_index = inputs[4];
  hstride_index = inputs[5];
  vstride_index = inputs[6];
  kw_index = inputs[7];
  kh_index = inputs[8];
  activation_index = inputs[9];
}

} // namespace Explicit

namespace Implicit
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 7 && outputCount == 1);

  ofm_index = outputs[0];

  // Each input should be interpreted as follows:
  //
  //  0 -> IFM Tensor Index
  //  1 -> Padding Code (ANEURALNETWORKS_PADDING_SAME or ANEURALNETWORKS_PADDING_VALID) Index
  //  2 -> Horizontal (over width) Stride Index
  //  3 -> Vertial (over height) Stride Index
  //  4 -> Filter Width Index
  //  5 -> Filter Height Index
  //  6 -> FuseCode (activation) Index
  ifm_index = inputs[0];
  padding_index = inputs[1];
  hstride_index = inputs[2];
  vstride_index = inputs[3];
  kw_index = inputs[4];
  kh_index = inputs[5];
  activation_index = inputs[6];
}

} // namespace Implicit
} // namespace MaxPool2D
} // namespace op
} // namespace tflite
} // namespace internal
