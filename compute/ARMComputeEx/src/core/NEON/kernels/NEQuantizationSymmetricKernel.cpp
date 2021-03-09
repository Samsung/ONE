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

#include "arm_compute/core/NEON/kernels/NEQuantizationSymmetricKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/INEKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/CPP/Validate.h"

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const ITensorInfo *scale_factor)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8_SIGNED);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scale_factor, 1, DataType::F16,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->num_dimensions() > 1);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->dimension(0) != input->dimension(1));

  return Status{};
}

inline float32x4x4_t load_value(const float *input_ptr)
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

inline float32x4_t round(const float32x4_t &fv)
{
  const float32x4_t point5_f32x4 = vdupq_n_f32(0.5f);
  const float32x4_t zero_f32x4 = vdupq_n_f32(0.0f);
  // If value < 0, mask = -1, else mask = 0
  int32x4_t mask_less_zero_ui32x4 = reinterpret_cast<int32x4_t>(vcltq_f32(fv, zero_f32x4));
  return vaddq_f32(fv, vaddq_f32(vcvtq_f32_s32(mask_less_zero_ui32x4), point5_f32x4));
}

inline int8x16_t vquantizeSymm(const float32x4x4_t &fv, float scale_factor_inv, int32_t max_scale)
{
  const float32x4_t vinvscale = vdupq_n_f32(scale_factor_inv);
  const int32x4_t vposend = vdupq_n_s32(max_scale);
  const int32x4_t vnagend = vdupq_n_s32(-max_scale);

  const int32x4x4_t rf = {{
#ifdef __aarch64__
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtnq_s32_f32(round(vmulq_f32(fv.val[0], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtnq_s32_f32(round(vmulq_f32(fv.val[1], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtnq_s32_f32(round(vmulq_f32(fv.val[2], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtnq_s32_f32(round(vmulq_f32(fv.val[3], vinvscale))))),
#else  //__aarch64__
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtq_s32_f32(round(vmulq_f32(fv.val[0], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtq_s32_f32(round(vmulq_f32(fv.val[1], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtq_s32_f32(round(vmulq_f32(fv.val[2], vinvscale))))),
    vminq_s32(vposend, vmaxq_s32(vnagend, vcvtq_s32_f32(round(vmulq_f32(fv.val[3], vinvscale))))),
#endif //__aarch64__
  }};
  const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
  const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
  return vcombine_s8(pa, pb);
}
} // namespace

NEQuantizationSymmetricKernel::NEQuantizationSymmetricKernel()
  : _input(nullptr), _output(nullptr), _scale_factor(nullptr)
{
}

void NEQuantizationSymmetricKernel::configure(const ITensor *input, ITensor *output,
                                              ITensor *scale_factor)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(
    validate_arguments(input->info(), output->info(), scale_factor->info()));

  _input = input;
  _output = output;
  _scale_factor = scale_factor;

  // Configure kernel window
  Window win_config = calculate_max_window(*input->info(), Steps());

  Coordinates coord;
  coord.set_num_dimensions(output->info()->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

  INEKernel::configure(win_config);
}

Status NEQuantizationSymmetricKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                               const ITensorInfo *scale_factor)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, scale_factor));

  return Status{};
}

template <typename T> void NEQuantizationSymmetricKernel::quantize(const Window &window)
{
  constexpr auto window_step = 16;
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());

#ifdef __aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP;
#endif //__aarch64__

  // Collapse window and reset first dimension to handle tail calculations manually
  // Support Only 2D input
  Window win_collapsed = window;
  Iterator input(_input, win_collapsed);
  Iterator output(_output, win_collapsed);
  const auto dim_x = _input->info()->dimension(0);
  win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
  execute_window_loop(
    win_collapsed,
    [&](const Coordinates &id) {
      const auto start = reinterpret_cast<const T *>(input.ptr());
      const auto min_max = std::minmax_element(start, start + dim_x);
      const auto int8_scale = 127;
      auto range = std::max(std::abs(*min_max.first), std::abs(*min_max.second));
      if (range == 0)
      {
        *reinterpret_cast<T *>(_scale_factor->ptr_to_element({id.y()})) = 1;
        range = 1;
      }
      else
      {
        *reinterpret_cast<T *>(_scale_factor->ptr_to_element({id.y()})) = range / int8_scale;
      }
      const auto scale_factor_inv = int8_scale / range;

      auto input_ptr = reinterpret_cast<const T *>(input.ptr());
      auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());
      int x = window_start_x;
      for (; x <= (window_end_x - window_step); x += window_step)
      {
        wrapper::vstore(&output_ptr[x],
                        vquantizeSymm(load_value(&input_ptr[x]), scale_factor_inv, int8_scale));
      }
      // Compute left-over elements
      for (; x < window_end_x; ++x)
      {
        int quantized = arm_compute::round(input_ptr[x] * scale_factor_inv, rounding_policy);
        quantized = std::min(int8_scale, std::max(quantized, -int8_scale));
        output_ptr[x] = static_cast<int8_t>(quantized);
      }
    },
    input, output);
}

void NEQuantizationSymmetricKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

  switch (_input->info()->data_type())
  {
    case DataType::F32:
      NEQuantizationSymmetricKernel::quantize<float>(window);
      break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    case DataType::F16:
      NEQuantizationSymmetricKernel::quantize<float16_t>(window);
      break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    default:
      ARM_COMPUTE_ERROR("Unsupported data type.");
  }
}
