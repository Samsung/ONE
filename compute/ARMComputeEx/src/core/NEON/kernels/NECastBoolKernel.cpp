/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NECastBoolKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/SaturateCast.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
  ARM_COMPUTE_RETURN_ERROR_ON(input == output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S8,
                                                       DataType::S16, DataType::U16, DataType::F16,
                                                       DataType::U32, DataType::S32, DataType::F32);

  // Validate in case of configured output
  if (output->total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  }

  return Status{};
}
} // namespace

NECastBoolKernel::NECastBoolKernel() : _input(nullptr), _output(nullptr) {}

void NECastBoolKernel::configure(const ITensor *input, ITensor *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  // Auto initialize output shape if not initialized (We can only auto-configure the shape, datatype
  // must be given)
  set_shape_if_empty(*output->info(), input->info()->tensor_shape());

  _input = input;
  _output = output;

  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

  // Configure kernel window
  Window win = calculate_max_window(*input->info(), Steps());
  Coordinates coord;
  coord.set_num_dimensions(output->info()->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

  ICPPKernel::configure(win);
}

Status NECastBoolKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
  return Status{};
}

void NECastBoolKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
  ARM_COMPUTE_ERROR_ON_NULLPTR(_input, _output);
  ARM_COMPUTE_ERROR_ON(_input == _output);

  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());
  const int window_step_x = 16;

  Window win{window};
  win.set(Window::DimX, Window::Dimension(0, 1, 1));

  Iterator input(_input, win);
  Iterator output(_output, win);

  const uint8_t true_val = 1;
  const uint8x8_t mask_bool = vdup_n_u8(true_val);

  switch (_output->info()->data_type())
  {
    case DataType::S8:
    {
      /* Conversion U8 -> S8 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              vst1q_s8(output_ptr + x,
                       vreinterpretq_s8_u8(vandq_u8(texels_u8, vdupq_n_u8(true_val))));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<int8_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
    case DataType::S16:
    {
      /* Up-conversion U8 -> S16 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              const int16x8x2_t texels = {
                  {vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_low_u8(texels_u8), mask_bool))),
                   vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_high_u8(texels_u8), mask_bool)))}};

              vst1q_s16(output_ptr + x, texels.val[0]);
              vst1q_s16(output_ptr + x + 8, texels.val[1]);
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<int32_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
    case DataType::S32:
    {
      /* Up-conversion U8 -> S32 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              const int16x8x2_t texels = {
                  {vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_low_u8(texels_u8), mask_bool))),
                   vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_high_u8(texels_u8), mask_bool)))}};

              vst1q_s32(output_ptr + x, vmovl_s16(vget_low_s16(texels.val[0])));
              vst1q_s32(output_ptr + x + 4, vmovl_s16(vget_high_s16(texels.val[0])));
              vst1q_s32(output_ptr + x + 8, vmovl_s16(vget_low_s16(texels.val[1])));
              vst1q_s32(output_ptr + x + 12, vmovl_s16(vget_high_s16(texels.val[1])));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<uint32_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
    case DataType::F32:
    {
      /* Up-conversion U8 -> F32 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              const int16x8x2_t texels = {
                  {vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_low_u8(texels_u8), mask_bool))),
                   vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_high_u8(texels_u8), mask_bool)))}};
              vst1q_f32(output_ptr + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[0]))));
              vst1q_f32(output_ptr + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[0]))));
              vst1q_f32(output_ptr + x + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[1]))));
              vst1q_f32(output_ptr + x + 12,
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[1]))));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              auto in = static_cast<uint32_t>(*(input_ptr + x) & true_val);
              *(output_ptr + x) = static_cast<float>(in);
            }
          },
          input, output);
      break;
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    case DataType::F16:
    {
      /* Up-conversion U8 -> F16 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              const int16x8x2_t texels = {
                  {vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_low_u8(texels_u8), mask_bool))),
                   vreinterpretq_s16_u16(vmovl_u8(vand_u8(vget_high_u8(texels_u8), mask_bool)))}};
              vst1q_f16(output_ptr + x, vcvtq_f16_s16(texels.val[0]));
              vst1q_f16(output_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<float16_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    case DataType::U8:
    {
      /* Conversion U8 -> S8 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              vst1q_u8(output_ptr + x, vandq_u8(texels_u8, vdupq_n_u8(true_val)));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<uint8_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
    case DataType::U16:
    {
      /* Up-conversion U8 -> U16 */
      execute_window_loop(
          win,
          [&](const Coordinates &) {
            const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<uint16_t *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
              const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

              const uint16x8x2_t texels = {{vmovl_u8(vand_u8(vget_low_u8(texels_u8), mask_bool)),
                                            vmovl_u8(vand_u8(vget_high_u8(texels_u8), mask_bool))}};

              vst1q_u16(output_ptr + x, texels.val[0]);
              vst1q_u16(output_ptr + x + 8, texels.val[1]);
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
              *(output_ptr + x) = static_cast<uint16_t>(*(input_ptr + x) & true_val);
            }
          },
          input, output);
      break;
    }
    default:
      ARM_COMPUTE_ERROR("Output data type not supported");
  }
}
