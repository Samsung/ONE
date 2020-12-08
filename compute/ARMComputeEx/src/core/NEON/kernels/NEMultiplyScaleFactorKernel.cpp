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

#include "arm_compute/core/NEON/kernels/NEMuliplyScaleFactorKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/CPP/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *scale_factor,
                          const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
  ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scale_factor, 1, DataType::F16,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->num_dimensions() > 1);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->dimension(0) != input->dimension(1));

  // Checks performed when output is configured
  if ((output->total_size() != 0))
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  }

  return Status{};
}

inline int32x4x4_t load_value(const int32_t *input_ptr)
{
  return {wrapper::vloadq(input_ptr), wrapper::vloadq(input_ptr + 4),
          wrapper::vloadq(input_ptr + 8), wrapper::vloadq(input_ptr + 12)};
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline const float32x4x4_t load_value(const float16_t *input_ptr)
{
  return {vcvt_f32_f16(wrapper::vload(input_ptr)), vcvt_f32_f16(wrapper::vload(input_ptr + 4)),
          vcvt_f32_f16(wrapper::vload(input_ptr + 8)),
          vcvt_f32_f16(wrapper::vload(input_ptr + 12))};
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename T> inline void store_result(T *ptr, const float32x4x4_t &v)
{
  ARM_COMPUTE_UNUSED(ptr, v);
}

template <> inline void store_result<float>(float *ptr, const float32x4x4_t &v)
{
  wrapper::vstore(ptr, v.val[0]);
  wrapper::vstore(ptr + 4, v.val[1]);
  wrapper::vstore(ptr + 8, v.val[2]);
  wrapper::vstore(ptr + 12, v.val[3]);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> inline void store_result<float16_t>(float16_t *ptr, const float32x4x4_t &v)
{
  wrapper::vstore(ptr, vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1])));
  wrapper::vstore(ptr + 8, vcombine_f16(vcvt_f16_f32(v.val[2]), vcvt_f16_f32(v.val[3])));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

inline float32x4x4_t multiply_scale_vec(const int32x4x4_t &iv, float scale)
{
  const float32x4_t vscale = vdupq_n_f32(scale);

  const float32x4x4_t ret = {{
    vmulq_f32(vcvtq_f32_s32(iv.val[0]), vscale),
    vmulq_f32(vcvtq_f32_s32(iv.val[1]), vscale),
    vmulq_f32(vcvtq_f32_s32(iv.val[2]), vscale),
    vmulq_f32(vcvtq_f32_s32(iv.val[3]), vscale),
  }};
  return ret;
}
} // namespace

NEMultiplyScaleFactorKernel::NEMultiplyScaleFactorKernel()
  : _input(nullptr), _scale_factor(nullptr), _output(nullptr), _multiplier(1.f)
{
}

void NEMultiplyScaleFactorKernel::configure(const ITensor *input, const ITensor *scale_factor,
                                            ITensor *output, float multiplier)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(
    validate_arguments(input->info(), scale_factor->info(), output->info()));

  _input = input;
  _scale_factor = scale_factor;
  _output = output;
  _multiplier = multiplier;

  // Configure kernel window
  Window win_config = calculate_max_window(*input->info(), Steps());

  Coordinates coord;
  coord.set_num_dimensions(output->info()->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

  INEKernel::configure(win_config);
}

Status NEMultiplyScaleFactorKernel::validate(const ITensorInfo *input,
                                             const ITensorInfo *scale_factor,
                                             const ITensorInfo *output, float multiplier)
{
  ARM_COMPUTE_UNUSED(multiplier);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, scale_factor, output));

  return Status{};
}

template <typename T> void NEMultiplyScaleFactorKernel::multiply(const Window &window)
{
  constexpr auto window_step = 16;
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());

  // Collapse window and reset first dimension to handle tail calculations manually
  // Support Only 2D input
  Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
  Iterator input(_input, win_collapsed);
  Iterator output(_output, win_collapsed);
  win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
  execute_window_loop(
    win_collapsed,
    [&](const Coordinates &id) {
      auto scale = *reinterpret_cast<T *>(_scale_factor->ptr_to_element({id.y()}));
      scale *= _multiplier;

      const auto input_ptr = reinterpret_cast<const int32_t *>(input.ptr());
      auto output_ptr = reinterpret_cast<T *>(output.ptr());
      int x = window_start_x;
      for (; x <= (window_end_x - window_step); x += window_step)
      {
        store_result<float>(&output_ptr[x], multiply_scale_vec(load_value(&input_ptr[x]), scale));
      }
      // Compute left-over elements
      for (; x < window_end_x; ++x)
      {
        output_ptr[x] = input_ptr[x] * scale;
      }
    },
    input, output);
}

void NEMultiplyScaleFactorKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

  switch (_output->info()->data_type())
  {
    case DataType::F32:
      NEMultiplyScaleFactorKernel::multiply<float>(window);
      break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    case DataType::F16:
      NEMultiplyScaleFactorKernel::multiply<float16_t>(window);
      break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    default:
      ARM_COMPUTE_ERROR("Unsupported data type.");
  }
}
