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
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/NEPReLUKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEElementwiseOperationFuncs.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;
namespace
{

/** Conditional element-wise operations */
enum class ConditionalOperation
{
  PRELU, /**< (x * y) for x < 0, x for x >= 0 */
};

template <ConditionalOperation op, typename ScalarType>
inline ScalarType elementwise_conditional_op_scalar(const ScalarType &a, const ScalarType &b)
{
  auto res = ScalarType(0);

  switch (op)
  {
    case ConditionalOperation::PRELU:
      res = a < 0 ? a * b : a;
      break;
    default:
      ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
  }
  return res;
}

template <ConditionalOperation op>
inline uint8_t elementwise_conditional_op_quantized_scalar(const float &a, const float &b,
                                                           QuantizationInfo qinfo)
{
  return quantize_qasymm8(elementwise_conditional_op_scalar<op>(a, b), qinfo,
                          RoundingPolicy::TO_NEAREST_UP);
}

template <ConditionalOperation op, typename VectorType>
inline VectorType elementwise_conditional_op(const VectorType &a, const VectorType &b)
{
  VectorType res = {0, 0, 0, 0};
  VectorType const_0 = {0, 0, 0, 0};

  switch (op)
  {
    case ConditionalOperation::PRELU:
      res = wrapper::vbsl(wrapper::vcgt(a, const_0), a, wrapper::vmul(a, b));
      ;
      break;
    default:
      ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
  }
  return res;
}

template <ConditionalOperation op>
inline float32x4x4_t elementwise_conditional_op(const float32x4x4_t &a, const float32x4x4_t &b)
{
  float32x4x4_t out = {{
      elementwise_conditional_op<op>(a.val[0], b.val[0]),
      elementwise_conditional_op<op>(a.val[1], b.val[1]),
      elementwise_conditional_op<op>(a.val[2], b.val[2]),
      elementwise_conditional_op<op>(a.val[3], b.val[3]),
  }};
  return out;
}

template <ConditionalOperation op, typename ScalarType, typename VectorType>
inline VectorType elementwise_conditional_op_broadcast(const VectorType &a,
                                                       const ScalarType &broadcast_value,
                                                       const bool reorder)
{
  VectorType broadcast_vector = wrapper::vdup_n(broadcast_value, wrapper::traits::vector_128_tag());
  return elementwise_conditional_op<op>(reorder ? broadcast_vector : a,
                                        reorder ? a : broadcast_vector);
}

template <ConditionalOperation op, typename ScalarType, typename VectorType>
inline int elementwise_conditional_op_loop(int window_start_x, int window_end_x, int window_step_x,
                                           const ScalarType *input1_ptr,
                                           const ScalarType *input2_ptr, ScalarType *output_ptr)
{
  int x = window_start_x;
  for (; x <= (window_end_x - window_step_x); x += window_step_x)
  {
    const auto a = wrapper::vloadq(input1_ptr + x);
    const auto b = wrapper::vloadq(input2_ptr + x);
    wrapper::vstore(output_ptr + x, elementwise_conditional_op<op>(a, b));
  }
  return x;
}

template <ConditionalOperation op>
inline int elementwise_conditional_op_quantized_loop(int window_start_x, int window_end_x,
                                                     int window_step_x, const uint8_t *input1_ptr,
                                                     const uint8_t *input2_ptr, uint8_t *output_ptr,
                                                     int32x4_t voffset1, int32x4_t voffset2,
                                                     float32x4_t vscale1, float32x4_t vscale2,
                                                     float32x4_t voffseto, float32x4_t invvscaleo)
{
  int x = window_start_x;
  for (; x <= (window_end_x - window_step_x); x += window_step_x)
  {
    // Get inputs and compute output
    const float32x4x4_t af = load_quantized(input1_ptr + x, voffset1, vscale1);
    const float32x4x4_t bf = load_quantized(input2_ptr + x, voffset2, vscale2);
    const float32x4x4_t rf = elementwise_conditional_op<op>(af, bf);
    store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
  }
  return x;
}

template <ConditionalOperation op, typename ScalarType, typename VectorType>
inline int elementwise_conditional_op_broadcast_loop(int window_start_x, int window_end_x,
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
                    elementwise_conditional_op_broadcast<op>(a, broadcast_value, reorder));
  }
  return x;
}

template <ConditionalOperation op>
inline int elementwise_conditional_op_quantized_broadcast_loop(
    int window_start_x, int window_end_x, int window_step_x, const uint8_t *non_broadcast_input_ptr,
    float32x4x4_t broadcast_vector, uint8_t *output_ptr, int32x4_t voffset_non_broadcast,
    float32x4_t vscale_non_broadcast, float32x4_t voffseto, float32x4_t invvscaleo, bool reorder)
{
  int x = window_start_x;
  for (; x <= (window_end_x - window_step_x); x += window_step_x)
  {
    const float32x4x4_t af =
        load_quantized(non_broadcast_input_ptr + x, voffset_non_broadcast, vscale_non_broadcast);
    const float32x4x4_t rf = elementwise_conditional_op<op>(reorder ? broadcast_vector : af,
                                                            reorder ? af : broadcast_vector);
    store_quantized(output_ptr + x, rf, voffseto, invvscaleo);
  }
  return x;
}

template <ConditionalOperation op, typename ScalarType, typename VectorType>
void elementwise_conditional_op(const ITensor *in1, const ITensor *in2, ITensor *out,
                                const Window &window)
{
  elementwise_op(in1, in2, out, window, &elementwise_conditional_op_scalar<op, ScalarType>,
                 &elementwise_conditional_op_broadcast_loop<op, ScalarType, VectorType>,
                 &elementwise_conditional_op_loop<op, ScalarType, VectorType>);
}

template <ConditionalOperation op>
void elementwise_conditional_op_quantized(const ITensor *in1, const ITensor *in2, ITensor *out,
                                          const Window &window)
{
  elementwise_op_quantized(in1, in2, out, window, &elementwise_conditional_op_quantized_scalar<op>,
                           &elementwise_conditional_op_quantized_broadcast_loop<op>,
                           &elementwise_conditional_op_quantized_loop<op>);
}
} // namespace

NEPReLUKernel::NEPReLUKernel() : _input(nullptr), _alpha(nullptr), _output(nullptr) {}

void NEPReLUKernel::configure(const ITensor *input, const ITensor *alpha, ITensor *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, alpha, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input->info(), *alpha->info(), *output->info()));

  // Configure kernel window
  const std::pair<TensorShape, ValidRegion> broadcast_pair =
      ITensorInfo::broadcast_shape_and_valid_region(*input->info(), *alpha->info());
  const TensorShape &out_shape = broadcast_pair.first;
  const ValidRegion &valid_region = broadcast_pair.second;

  // Auto initialize output if not initialized
  auto_init_if_empty(*output->info(), out_shape, 1, input->info()->data_type());

  Window win = calculate_max_window(valid_region);

  _input = input;
  _alpha = alpha;
  _output = output;
  INEKernel::configure(win);
}

void NEPReLUKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  if (_input->info()->data_type() == DataType::F32)
  {
    elementwise_conditional_op<ConditionalOperation::PRELU, float, float32x4_t>(_input, _alpha,
                                                                                _output, window);
  }
  else if (_input->info()->data_type() == DataType::QASYMM8)
  {
    elementwise_conditional_op_quantized<ConditionalOperation::PRELU>(_input, _alpha, _output,
                                                                      window);
  }
  else
  {
    ARM_COMPUTE_ERROR("Wrong Type");
  }
}

Status NEPReLUKernel::validate_arguments(const ITensorInfo &input, const ITensorInfo &alpha,
                                         const ITensorInfo &output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &alpha, &output);

  const TensorShape out_shape =
      TensorShape::broadcast_shape(input.tensor_shape(), alpha.tensor_shape());

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0,
                                  "Inputs are not broadcast compatible");

  // Checks performed when output is configured
  if (output.total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
        "Wrong shape for output");
  }

  return Status{};
}

Status NEPReLUKernel::validate(const ITensorInfo *input, const ITensorInfo *alpha,
                               const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, alpha, output);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input, *alpha, *output));

  return Status{};
}
