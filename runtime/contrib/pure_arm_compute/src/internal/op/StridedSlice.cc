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

#include "internal/op/StridedSlice.h"
#include "internal/op/NodeVisitor.h"

#include <cassert>

namespace internal
{
namespace tflite
{
namespace op
{
namespace StridedSlice
{

void Node::accept(NodeVisitor &&v) const { v.visit(*this); }

} // namespace StridedSlice
} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace op
{
namespace StridedSlice
{

Param::Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount,
             const uint32_t *outputs)
{
  assert(inputCount == 7 && outputCount == 1);

  outputData_index = outputs[0];

  // Each input should be interpreted as follows:
  //
  //  0 -> An n-D tensor, specifying the tensor to be sliced.
  //  1 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the starts of
  //       the dimensions of the input tensor to be sliced. The length must be
  //       of rank(input0).
  //  2 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the ends of
  //       the dimensions of the input tensor to be sliced. The length must be
  //       of rank(input0).
  //  3 -> A 1-D Tensor of {@link ANEURALNETWORKS_TENSOR_INT32}, the strides of
  //       the dimensions of the input tensor to be sliced. The length must be
  //       of rank(input0).
  //  4 -> An {@link ANEURALNETWORKS_INT32} scalar, begin_mask. If the ith bit
  //       of begin_mask is set, begin[i] is ignored and the fullest possible
  //       range in that dimension is used instead.
  //  5 -> An {@link ANEURALNETWORKS_INT32} scalar, end_mask. If the ith bit of
  //       end_mask is set, end[i] is ignored and the fullest possible range in
  //       that dimension is used instead.
  //  6 -> An {@link ANEURALNETWORKS_INT32} scalar, shrink_axis_mask. An int32
  //       mask. If the ith bit of shrink_axis_mask is set, it implies that the
  //       ith specification shrinks the dimensionality by 1. A slice of size 1
  //       starting from begin[i] in the dimension must be preserved.
  inputData_index = inputs[0];
  startData_index = inputs[1];
  endData_index = inputs[2];
  stridesData_index = inputs[3];
  beginMask_index = inputs[4];
  endMask_index = inputs[5];
  shrinkAxisMask_index = inputs[6];
}

} // namespace StridedSlice
} // namespace op
} // namespace tflite
} // namespace internal
