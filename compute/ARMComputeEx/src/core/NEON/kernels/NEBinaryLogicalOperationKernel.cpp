/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2018-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/core/NEON/kernels/NEBinaryLogicalOperationKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/NEON/NEElementwiseOperationFuncs.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>
#include <map>
#include <string>

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace arm_compute
{

template <BinaryLogicalOperation op, typename ScalarType>
inline ScalarType elementwise_logic_op_scalar(const ScalarType &a, const ScalarType &b)
{
  auto res = ScalarType(0);

  switch (op)
  {
    case BinaryLogicalOperation::AND:
      res = a & b;
      break;
    case BinaryLogicalOperation::OR:
      res = a | b;
      break;
    default:
      ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
  }
  return res;
}

template <BinaryLogicalOperation op, typename VectorType>
inline VectorType elementwise_logic_op(const VectorType &a, const VectorType &b)
{
  VectorType res = {0, 0, 0, 0};

  switch (op)
  {
    case BinaryLogicalOperation::AND:
      res = wrapper::vand(a, b);
      break;
    case BinaryLogicalOperation::OR:
      res = wrapper::vorr(a, b);
      break;
    default:
      ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
  }
  return res;
}

template <BinaryLogicalOperation op>
inline uint8x16x4_t elementwise_logic_op(const uint8x16x4_t &a, const uint8x16x4_t &b)
{
  uint8x16x4_t out = {{
    elementwise_logic_op<op>(a.val[0], b.val[0]),
    elementwise_logic_op<op>(a.val[1], b.val[1]),
    elementwise_logic_op<op>(a.val[2], b.val[2]),
    elementwise_logic_op<op>(a.val[3], b.val[3]),
  }};
  return out;
}

template <BinaryLogicalOperation op, typename ScalarType, typename VectorType>
inline VectorType elementwise_logic_op_broadcast(const VectorType &a,
                                                 const ScalarType &broadcast_value,
                                                 const bool reorder)
{
  VectorType broadcast_vector = wrapper::vdup_n(broadcast_value, wrapper::traits::vector_128_tag());
  return elementwise_logic_op<op>(reorder ? broadcast_vector : a, reorder ? a : broadcast_vector);
}

template <BinaryLogicalOperation op, typename ScalarType, typename VectorType>
inline int elementwise_logic_op_loop(int window_start_x, int window_end_x, int window_step_x,
                                     const ScalarType *input1_ptr, const ScalarType *input2_ptr,
                                     ScalarType *output_ptr)
{
  int x = window_start_x;
  for (; x <= (window_end_x - window_step_x); x += window_step_x)
  {
    const auto a = wrapper::vloadq(input1_ptr + x);
    const auto b = wrapper::vloadq(input2_ptr + x);
    wrapper::vstore(output_ptr + x, elementwise_logic_op<op>(a, b));
  }
  return x;
}

template <BinaryLogicalOperation op, typename ScalarType, typename VectorType>
inline int elementwise_logic_op_broadcast_loop(int window_start_x, int window_end_x,
                                               int window_step_x,
                                               const ScalarType *non_broadcast_input_ptr,
                                               const ScalarType &broadcast_value,
                                               ScalarType *output_ptr, const bool reorder)
{
  int x = window_start_x;
  for (; x <= (window_end_x - window_step_x); x += window_step_x)
  {
    const auto a = wrapper::vloadq((non_broadcast_input_ptr + x));
    wrapper::vstore(output_ptr + x,
                    elementwise_logic_op_broadcast<op>(a, broadcast_value, reorder));
  }
  return x;
}

template <BinaryLogicalOperation op, typename ScalarType, typename VectorType>
void elementwise_logic_op(const ITensor *in1, const ITensor *in2, ITensor *out,
                          const Window &window)
{
  elementwise_op(in1, in2, out, window, &elementwise_logic_op_scalar<op, ScalarType>,
                 &elementwise_logic_op_broadcast_loop<op, ScalarType, VectorType>,
                 &elementwise_logic_op_loop<op, ScalarType, VectorType>);
}

std::function<void(const ITensor *, const ITensor *, ITensor *, const Window &)> configure_func(
  const ITensor *input1, const ITensor *input2, ITensor *output,
  std::map<std::string, NEElementwiseOperationKernel::ElementwiseFunction *> map_function)
{
  std::string function_to_call("op_");
  function_to_call += string_from_data_type(input1->info()->data_type()) + "_";
  function_to_call += string_from_data_type(input2->info()->data_type()) + "_";
  function_to_call += string_from_data_type(output->info()->data_type());

  auto it = map_function.find(function_to_call);

  if (it != map_function.end())
  {
    auto func = it->second;
    return [func](const ITensor *input1, const ITensor *input2, ITensor *output,
                  const Window &window) { func(input1, input2, output, window); };
  }
  return nullptr;
}

template <BinaryLogicalOperation op>
std::function<void(const ITensor *, const ITensor *, ITensor *, const Window &)>
configure_logic_func(const ITensor *input1, const ITensor *input2, ITensor *output)
{
  static std::map<std::string, NEElementwiseOperationKernel::ElementwiseFunction *> map_function = {
    {"op_U8_U8_U8", &elementwise_logic_op<op, uint8_t, uint8x16_t>},
    {"op_QASYMM8_QASYMM8_QASYMM8", &elementwise_logic_op<op, uint8_t, uint8x16_t>}};

  return configure_func(input1, input2, output, map_function);
}

void NEBinaryLogicalOperationKernel::configure(BinaryLogicalOperation op, const ITensor *input1,
                                               const ITensor *input2, ITensor *output)
{
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1->info(), *input2->info(), *output->info()));
  configure_common(input1, input2, output);
  switch (op)
  {
    case BinaryLogicalOperation::AND:
      _function = configure_logic_func<BinaryLogicalOperation::AND>(input1, input2, output);
      break;
    case BinaryLogicalOperation::OR:
      _function = configure_logic_func<BinaryLogicalOperation::OR>(input1, input2, output);
      break;
    default:
      ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
  }
}

Status NEBinaryLogicalOperationKernel::validate_arguments(const ITensorInfo &input1,
                                                          const ITensorInfo &input2,
                                                          const ITensorInfo &output)
{
  // Validate in case of configured output
  if (output.total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8,
                                                         DataType::QASYMM8);
  }
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);

  const TensorShape out_shape =
    TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0,
                                  "Inputs are not broadcast compatible");

  // Validate in case of configured output
  if (output.total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
      detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
      "Wrong shape for output");
  }

  return Status{};
}

Status NEBinaryLogicalOperationKernel::validate(BinaryLogicalOperation op,
                                                const ITensorInfo *input1,
                                                const ITensorInfo *input2,
                                                const ITensorInfo *output)
{
  ARM_COMPUTE_UNUSED(op);
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
  return Status{};
}

} // namespace arm_compute
