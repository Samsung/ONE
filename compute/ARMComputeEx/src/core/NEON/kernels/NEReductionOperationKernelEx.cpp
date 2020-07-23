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

#include "arm_compute/core/NEON/kernels/NEReductionOperationKernelEx.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace
{
// Helper function to calculate the minimum value of the input vector. All the elements in the
// output vector contain the min value.
float32x2_t calculate_min(float32x4_t in)
{
  auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
  return wrapper::vpmin(pmin, pmin);
}

// Helper function to calculate the maximum value of the input vector. All the elements in the
// output vector contain the max value.
float32x2_t calculate_max(float32x4_t in)
{
  auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
  return wrapper::vpmax(pmax, pmax);
}
// Helper function to calculate the minimum value of the input vector. All the elements in the
// output vector contain the min value.
int32x2_t calculate_min(int32x4_t in)
{
  auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
  return wrapper::vpmin(pmin, pmin);
}

// Helper function to calculate the maximum value of the input vector. All the elements in the
// output vector contain the max value.
int32x2_t calculate_max(int32x4_t in)
{
  auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
  return wrapper::vpmax(pmax, pmax);
}

// Helper function to calculate the minimum value of the input vector. All the elements in the
// output vector contain the min value.
inline uint8x8_t calculate_min(uint8x16_t in)
{
  auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
  pmin = wrapper::vpmin(pmin, pmin);
  pmin = wrapper::vpmin(pmin, pmin);
  return wrapper::vpmin(pmin, pmin);
}
// Helper function to calculate the maximum value of the input vector. All the elements in the
// output vector contain the max value.
inline uint8x8_t calculate_max(uint8x16_t in)
{
  auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
  pmax = wrapper::vpmax(pmax, pmax);
  pmax = wrapper::vpmax(pmax, pmax);
  return wrapper::vpmax(pmax, pmax);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// Helper function to calculate the minimum value of the input vector. All the elements in the
// output vector contain the min value.
inline float16x4_t calculate_min(float16x8_t in)
{
  auto pmin = wrapper::vpmin(wrapper::vgethigh(in), wrapper::vgetlow(in));
  pmin = wrapper::vpmin(pmin, pmin);
  return wrapper::vpmin(pmin, pmin);
}
// Helper function to calculate the maximum value of the input vector. All the elements in the
// output vector contain the max value.
inline float16x4_t calculate_max(float16x8_t in)
{
  auto pmax = wrapper::vpmax(wrapper::vgethigh(in), wrapper::vgetlow(in));
  pmax = wrapper::vpmax(pmax, pmax);
  return wrapper::vpmax(pmax, pmax);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <class F> class Reducer
{
public:
  static void reduceX(const Window &window, const ITensor *input, ITensor *output, F f,
                      const ReduceOperation op)
  {
    // Set out window
    Window out_window(window);
    out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

    // Get first input and output slices
    Window in_slice = window.first_slice_window_1D();
    Window out_slice = out_window.first_slice_window_1D();

    do
    {
      Iterator in(input, in_slice);
      Iterator out(output, out_slice);

      f(in, out, in_slice, out_slice, *input->info(), op);
    } while (window.slide_window_slice_1D(in_slice) && out_window.slide_window_slice_1D(out_slice));
  }
  static void reduceY(const Window &window, const ITensor *input, ITensor *output, F f,
                      const ReduceOperation op)
  {
    // Set in window
    Window in_window(window);
    Window out_window(window);

    in_window.set(Window::DimY, Window::Dimension(0, 1, 1));
    out_window.set(Window::DimY, Window::Dimension(0, output->info()->dimension(1),
                                                   output->info()->dimension(1)));

    // Get first input and output slices
    Window in_slice = in_window.first_slice_window_2D();
    Window out_slice = out_window.first_slice_window_2D();

    do
    {
      Iterator in(input, in_slice);
      Iterator out(output, out_slice);

      f(in, out, in_slice, out_slice, *input->info(), 1, op);
    } while (in_window.slide_window_slice_2D(in_slice) &&
             out_window.slide_window_slice_2D(out_slice));
  }
  static void reduceZ(const Window &window, const ITensor *input, ITensor *output, F f,
                      const ReduceOperation op)
  {
    // Set in window
    Window in_window(window);
    Window out_window(window);

    in_window.set(Window::DimZ, Window::Dimension(0, 1, 1));
    out_window.set(Window::DimZ, Window::Dimension(0, output->info()->dimension(2),
                                                   output->info()->dimension(2)));

    // Get first input and output slices
    Window in_slice = in_window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_3D();

    do
    {
      Iterator in(input, in_slice);
      Iterator out(output, out_slice);

      f(in, out, in_slice, out_slice, *input->info(), 2, op);
    } while (in_window.slide_window_slice_3D(in_slice) &&
             out_window.slide_window_slice_3D(out_slice));
  }
  static void reduceW(const Window &window, const ITensor *input, ITensor *output, F f,
                      const ReduceOperation op)
  {
    // Set in/out window
    Window in_window(window);
    Window out_window(window);

    in_window.set(3, Window::Dimension(0, 1, 1));
    out_window.set(3, Window::Dimension(0, 1, 1));

    // Get first input and output slices
    Window in_slice = in_window.first_slice_window_4D();
    Window out_slice = out_window.first_slice_window_4D();

    do
    {
      Iterator in(input, in_slice);
      Iterator out(output, out_slice);

      f(in, out, in_slice, out_slice, *input->info(), 3, op);
    } while (in_window.slide_window_slice_4D(in_slice) &&
             out_window.slide_window_slice_4D(out_slice));
  }
};

template <typename T, int S> struct RedOpX
{
  /** NEON vector tag type. */
  using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

  inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice,
                         const TensorInfo &in_info, const ReduceOperation op)
  {
    ARM_COMPUTE_UNUSED(out_slice);
    ARM_COMPUTE_UNUSED(in_info);
    auto init_res_value = static_cast<T>(0.f);
    switch (op)
    {
      case ReduceOperation::MIN:
      case ReduceOperation::MAX:
      {
        init_res_value = *reinterpret_cast<T *>(input.ptr());
        break;
      }
      default:
        break;
    }
    auto vec_res_value = wrapper::vdup_n(init_res_value, ExactTagType{});

    execute_window_loop(
        in_slice,
        [&](const Coordinates &) {
          const auto in_ptr = reinterpret_cast<const T *>(input.ptr());
          const auto vec_elements = wrapper::vloadq(in_ptr);

          switch (op)
          {
            case ReduceOperation::MIN:
            {
              vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
              break;
            }
            case ReduceOperation::MAX:
            {
              vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Not supported");
          }
        },
        input);

    switch (op)
    {
      case ReduceOperation::MIN:
      {
        *(reinterpret_cast<T *>(output.ptr())) = wrapper::vgetlane(calculate_min(vec_res_value), 0);
        break;
      }
      case ReduceOperation::MAX:
      {
        *(reinterpret_cast<T *>(output.ptr())) = wrapper::vgetlane(calculate_max(vec_res_value), 0);
        break;
      }
      default:
        ARM_COMPUTE_ERROR("Not supported");
    }
  }
};

struct RedOpX_qasymm8
{
  inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice,
                         const TensorInfo &in_info, const ReduceOperation op)
  {
    ARM_COMPUTE_UNUSED(out_slice);
    ARM_COMPUTE_UNUSED(in_info);

    uint8x16_t vec_res_value = {0};

    if (op == ReduceOperation::MIN || op == ReduceOperation::MAX)
    {
      vec_res_value = wrapper::vdup_n(*input.ptr(), wrapper::traits::vector_128_tag{});
    }

    execute_window_loop(
        in_slice,
        [&](const Coordinates &) {
          const auto vec_elements = wrapper::vloadq(input.ptr());
          switch (op)
          {
            case ReduceOperation::MIN:
            {
              vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
              break;
            }
            case ReduceOperation::MAX:
            {
              vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Not supported");
          }
        },
        input);

    switch (op)
    {
      case ReduceOperation::MIN:
      {
        *(output.ptr()) = static_cast<uint8_t>(wrapper::vgetlane(calculate_min(vec_res_value), 0));
        break;
      }
      case ReduceOperation::MAX:
      {
        *(output.ptr()) = static_cast<uint8_t>(wrapper::vgetlane(calculate_max(vec_res_value), 0));
        break;
      }
      default:
      {
        ARM_COMPUTE_ERROR("Not supported");
      }
    }
  }
};

template <typename T, int S> struct RedOpYZW
{
  /** NEON vector tag type. */
  using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;
  using neon_vector = typename wrapper::traits::neon_vector<T, S>::type;

  inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice,
                         const TensorInfo &in_info, int axis, const ReduceOperation op)
  {
    ARM_COMPUTE_UNUSED(out_slice);

    execute_window_loop(
        in_slice,
        [&](const Coordinates &) {
          neon_vector vec_res_value = {0};
          switch (op)
          {
            case ReduceOperation::MIN:
            case ReduceOperation::MAX:
            {
              vec_res_value = wrapper::vloadq(reinterpret_cast<T *>(input.ptr()));
              break;
            }
            default:
            {
              vec_res_value = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
              break;
            }
          }

          for (unsigned int dim = 0; dim < in_info.dimension(axis); ++dim)
          {
            T *in_ptr;
            switch (axis)
            {
              case 1:
                in_ptr = reinterpret_cast<T *>(
                    input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, dim)));
                break;
              case 2:
                in_ptr = reinterpret_cast<T *>(
                    input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, dim)));
                break;
              case 3:
                in_ptr = reinterpret_cast<T *>(
                    input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, 0, dim)));
                break;
              default:
                ARM_COMPUTE_ERROR("Not supported");
            }
            const auto vec_elements = wrapper::vloadq(in_ptr);

            switch (op)
            {
              case ReduceOperation::MIN:
              {
                vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                break;
              }
              case ReduceOperation::MAX:
              {
                vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                break;
              }
              default:
                ARM_COMPUTE_ERROR("Not supported");
            }
          }
          wrapper::vstore(reinterpret_cast<T *>(output.ptr()), vec_res_value);
        },
        input, output);
  }
};

struct RedOpYZW_qasymm8
{
  inline void operator()(Iterator &input, Iterator &output, Window &in_slice, Window &out_slice,
                         const TensorInfo &in_info, int axis, const ReduceOperation op)
  {
    ARM_COMPUTE_UNUSED(out_slice);

    execute_window_loop(
        in_slice,
        [&](const Coordinates &) {
          auto vec_res_value = wrapper::vloadq(input.ptr());

          for (unsigned int index_dim = 0; index_dim < in_info.dimension(axis); ++index_dim)
          {
            uint8_t *in_ptr;
            switch (axis)
            {
              case 1:
                in_ptr = input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, index_dim));
                break;
              case 2:
                in_ptr =
                    input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, index_dim));
                break;
              case 3:
                in_ptr =
                    input.ptr() + in_info.offset_element_in_bytes(Coordinates(0, 0, 0, index_dim));
                break;
              default:
                ARM_COMPUTE_ERROR("Not supported");
            }
            const auto vec_elements = wrapper::vloadq(in_ptr);

            switch (op)
            {
              case ReduceOperation::MIN:
              {
                vec_res_value = wrapper::vmin(vec_elements, vec_res_value);
                break;
              }
              case ReduceOperation::MAX:
              {
                vec_res_value = wrapper::vmax(vec_elements, vec_res_value);
                break;
              }
              default:
                ARM_COMPUTE_ERROR("Not supported");
            }
          }
          wrapper::vstore(reinterpret_cast<uint8_t *>(output.ptr()), vec_res_value);
        },
        input, output);
  }
};

void reduce_op(const Window &window, const ITensor *input, ITensor *output, unsigned int axis,
               const ReduceOperation op)
{
  const bool is_complex = (input->info()->num_channels() == 2);
  if (is_complex)
  {
    ARM_COMPUTE_ERROR("Not supported");
  }

  switch (axis)
  {
    case 0:
      switch (input->info()->data_type())
      {
        case DataType::QASYMM8:
          return Reducer<RedOpX_qasymm8>::reduceX(window, input, output, RedOpX_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
          return Reducer<RedOpX<float16_t, 8>>::reduceX(window, input, output,
                                                        RedOpX<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
          return Reducer<RedOpX<float, 4>>::reduceX(window, input, output, RedOpX<float, 4>(), op);
        case DataType::S32:
          return Reducer<RedOpX<int32_t, 4>>::reduceX(window, input, output, RedOpX<int32_t, 4>(),
                                                      op);
        default:
          ARM_COMPUTE_ERROR("Not supported");
      }
    case 1:
      switch (input->info()->data_type())
      {
        case DataType::QASYMM8:
          return Reducer<RedOpYZW_qasymm8>::reduceY(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
          return Reducer<RedOpYZW<float16_t, 8>>::reduceY(window, input, output,
                                                          RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
          return Reducer<RedOpYZW<float, 4>>::reduceY(window, input, output, RedOpYZW<float, 4>(),
                                                      op);
        case DataType::S32:
          return Reducer<RedOpYZW<int32_t, 4>>::reduceY(window, input, output,
                                                        RedOpYZW<int32_t, 4>(), op);
        default:
          ARM_COMPUTE_ERROR("Not supported");
      }
    case 2:
      switch (input->info()->data_type())
      {
        case DataType::QASYMM8:
          return Reducer<RedOpYZW_qasymm8>::reduceZ(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
          return Reducer<RedOpYZW<float16_t, 8>>::reduceZ(window, input, output,
                                                          RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
          return Reducer<RedOpYZW<float, 4>>::reduceZ(window, input, output, RedOpYZW<float, 4>(),
                                                      op);
        case DataType::S32:
          return Reducer<RedOpYZW<int32_t, 4>>::reduceZ(window, input, output,
                                                        RedOpYZW<int32_t, 4>(), op);
        default:
          ARM_COMPUTE_ERROR("Not supported");
      }
    case 3:
      switch (input->info()->data_type())
      {
        case DataType::QASYMM8:
          return Reducer<RedOpYZW_qasymm8>::reduceW(window, input, output, RedOpYZW_qasymm8(), op);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
          return Reducer<RedOpYZW<float16_t, 8>>::reduceW(window, input, output,
                                                          RedOpYZW<float16_t, 8>(), op);
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
          return Reducer<RedOpYZW<float, 4>>::reduceW(window, input, output, RedOpYZW<float, 4>(),
                                                      op);
        case DataType::S32:
          return Reducer<RedOpYZW<int32_t, 4>>::reduceW(window, input, output,
                                                        RedOpYZW<int32_t, 4>(), op);
        default:
          ARM_COMPUTE_ERROR("Not supported");
      }
    default:
      ARM_COMPUTE_ERROR("Unsupported reduction axis");
  }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis,
                          ReduceOperation op)
{
  ARM_COMPUTE_UNUSED(op);

  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

  if (input->num_channels() == 1)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::S32,
                                                         DataType::F16, DataType::F32);
  }
  else
  {
    ARM_COMPUTE_RETURN_ERROR_MSG("Not support complex");
  }

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions,
                                  "Reduction axis greater than max number of dimensions");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() != output->num_channels());

    const TensorShape output_shape =
        arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);
    const TensorInfo tensor_info_reshaped = input->clone()->set_tensor_shape(output_shape);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_reshaped);
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output,
                                                         unsigned int axis, ReduceOperation op)
{
  ARM_COMPUTE_UNUSED(op);

  // Calculate output shape and set if empty
  const TensorShape output_shape =
      arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);

  // Output auto initialization if not yet initialized
  DataType output_data_type = input->data_type();
  auto_init_if_empty(*output, input->clone()
                                  ->set_tensor_shape(output_shape)
                                  .set_data_type(output_data_type)
                                  .reset_padding()
                                  .set_is_resizable(true));

  unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->data_type());

  // Configure kernel window
  Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
  AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

  bool window_changed = update_window_and_padding(win, input_access, output_access);
  output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

  Status err = (window_changed)
                   ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!")
                   : Status{};

  return std::make_tuple(err, win);
}
} // namespace

NEReductionOperationKernelEx::NEReductionOperationKernelEx()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReduceOperation::MAX),
      _border_size()
{
}

BorderSize NEReductionOperationKernelEx::border_size() const { return _border_size; }

void NEReductionOperationKernelEx::configure(const ITensor *input, ITensor *output,
                                             unsigned int axis, ReduceOperation op)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

  unsigned int num_elems_processed_per_iteration =
      16 / data_size_from_type(input->info()->data_type());

  _input = input;
  _output = output;
  _border_size =
      (axis == 0)
          ? BorderSize(0,
                       num_elems_processed_per_iteration -
                           (input->info()->dimension(0) % num_elems_processed_per_iteration),
                       0, 0)
          : BorderSize();
  _op = op;
  _reduction_axis = axis;

  // Configure kernel window
  auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis, op);

  ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

  INEKernel::configure(std::get<1>(win_config));
}

Status NEReductionOperationKernelEx::validate(const ITensorInfo *input, const ITensorInfo *output,
                                              unsigned int axis, ReduceOperation op)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
  ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(
      validate_and_configure_window(input->clone().get(), output->clone().get(), axis, op)));

  return Status{};
}

void NEReductionOperationKernelEx::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

  reduce_op(window, _input, _output, _reduction_axis, _op);
}
} // namespace arm_compute
