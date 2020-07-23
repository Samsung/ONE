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
 * Copyright (c) 2016-2018 ARM Limited.
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

#include "arm_compute/core/NEON/NEElementwiseOperationFuncs.h"

#include <algorithm>
#include "arm_compute/core/Types.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"

namespace
{
void store_quantized_int32(uint8_t *output_ptr, const int32x4x4_t &out)
{
  const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[0]), vqmovn_s32(out.val[1])));
  const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(out.val[2]), vqmovn_s32(out.val[3])));
  vst1q_u8(output_ptr, vcombine_u8(pa, pb));
}

using namespace arm_compute;
template <typename InputScalarType, typename OutputScalarType, typename InputVectorType>
void elementwise_op_templ(
    const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
    OutputScalarType (*scalar_func)(const InputScalarType &, const InputScalarType &),
    int (*broadcast_func)(int, int, int, const InputScalarType *, const InputScalarType &,
                          OutputScalarType *, const bool),
    int (*neon_func)(int, int, int, const InputScalarType *, const InputScalarType *,
                     OutputScalarType *))
{
  // Create input windows
  Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
  Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

  // Clear X Dimension on execution window as we handle manually
  Window win = window;
  win.set(Window::DimX, Window::Dimension(0, 1, 1));

  const int window_step_x = std::min(16 / static_cast<int>(sizeof(OutputScalarType)), 8);
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());
  const bool is_broadcast_across_x = (input1_win.x().step() == 0) || (input2_win.x().step() == 0);

  if (is_broadcast_across_x)
  {
    const bool is_broadcast_input_2 = input2_win.x().step() == 0;
    Window broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
    Window non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
    const ITensor *broadcast_tensor = is_broadcast_input_2 ? in2 : in1;
    const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

    // Clear X Dimension on execution window as we handle manually
    non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator broadcast_input(broadcast_tensor, broadcast_win);
    Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
    Iterator output(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &) {
          auto output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
          const auto non_broadcast_input_ptr =
              reinterpret_cast<const InputScalarType *>(non_broadcast_input.ptr());
          const InputScalarType broadcast_value =
              *reinterpret_cast<const InputScalarType *>(broadcast_input.ptr());

          int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x,
                                    non_broadcast_input_ptr, broadcast_value, output_ptr,
                                    !is_broadcast_input_2);
          for (; x < window_end_x; ++x)
          {
            const auto a = *(non_broadcast_input_ptr + x);
            *(output_ptr + x) = (*scalar_func)(!is_broadcast_input_2 ? broadcast_value : a,
                                               !is_broadcast_input_2 ? a : broadcast_value);
          }
        },
        broadcast_input, non_broadcast_input, output);
  }
  else
  {
    // Clear X Dimension on execution window as we handle manually
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &) {
          auto output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
          const auto input1_ptr = reinterpret_cast<const InputScalarType *>(input1.ptr());
          const auto input2_ptr = reinterpret_cast<const InputScalarType *>(input2.ptr());

          int x = (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr,
                               output_ptr);
          for (; x < window_end_x; ++x)
          {
            const auto a = *(input1_ptr + x);
            const auto b = *(input2_ptr + x);
            *(output_ptr + x) = (*scalar_func)(a, b);
          }
        },
        input1, input2, output);
  }
}

} // namespace

namespace arm_compute
{

float32x4x4_t load_quantized(const uint8_t *input1_ptr, const int32x4_t &offset,
                             const float32x4_t &scale)
{
  qasymm8x16_t x = vld1q_u8(input1_ptr);
  const float32x4x4_t out = {{
      vmulq_f32(
          vcvtq_f32_s32(vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(x))))), offset)),
          scale),
      vmulq_f32(
          vcvtq_f32_s32(vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(x))))), offset)),
          scale),
      vmulq_f32(
          vcvtq_f32_s32(vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(x))))), offset)),
          scale),
      vmulq_f32(
          vcvtq_f32_s32(vsubq_s32(
              vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(x))))), offset)),
          scale),
  }};
  return out;
}

void store_quantized(uint8_t *output_ptr, const float32x4x4_t &rf, const float32x4_t &offset,
                     const float32x4_t &invscale)
{
  int32x4x4_t out = {{
      vcvtq_s32_f32(vmlaq_f32(offset, rf.val[0], invscale)),
      vcvtq_s32_f32(vmlaq_f32(offset, rf.val[1], invscale)),
      vcvtq_s32_f32(vmlaq_f32(offset, rf.val[2], invscale)),
      vcvtq_s32_f32(vmlaq_f32(offset, rf.val[3], invscale)),
  }};
  store_quantized_int32(output_ptr, out);
}

float32x4x4_t dup_quantized(uint8_t broadcast_value, int offset, float scale)
{
  const qasymm8x16_t broadcast_value_vec = vdupq_n_u8(broadcast_value);
  const int32x4_t voffset = vdupq_n_s32(offset);
  const float32x4_t vscale = vdupq_n_f32(scale);

  const float32x4x4_t broadcast_vector = {{
      vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(
                                            vmovl_u8(vget_low_u8(broadcast_value_vec))))),
                                        voffset)),
                vscale),
      vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(
                                            vmovl_u8(vget_low_u8(broadcast_value_vec))))),
                                        voffset)),
                vscale),
      vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(
                                            vmovl_u8(vget_high_u8(broadcast_value_vec))))),
                                        voffset)),
                vscale),
      vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(
                                            vmovl_u8(vget_high_u8(broadcast_value_vec))))),
                                        voffset)),
                vscale),
  }};
  return broadcast_vector;
}

void elementwise_op_quantized(
    const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
    uint8_t (*scalar_func)(const float &, const float &, QuantizationInfo),
    int (*broadcast_func)(int, int, int, const uint8_t *, float32x4x4_t, uint8_t *, int32x4_t,
                          float32x4_t, float32x4_t, float32x4_t, const bool),
    int (*neon_func)(int, int, int, const uint8_t *, const uint8_t *, uint8_t *, int32x4_t,
                     int32x4_t, float32x4_t, float32x4_t, float32x4_t, float32x4_t))
{
  // Create input windows
  Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
  Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

  // Clear X Dimension on execution window as we handle manually
  Window win = window;
  win.set(Window::DimX, Window::Dimension(0, 1, 1));

  const int window_step_x = 16;
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());
  const bool is_broadcast_across_x = (input1_win.x().step() == 0) || (input2_win.x().step() == 0);

  UniformQuantizationInfo qinfo = out->info()->quantization_info().uniform();
  const float output_scale = qinfo.scale;
  const int output_offset = qinfo.offset;

  // Output quantization info (add 0.5 to round toward the nearest integer - 0.5 rounds away from
  // zero)
  const float32x4_t voffseto = vdupq_n_f32(output_offset + 0.5f);
  const float32x4_t invvscaleo = vdupq_n_f32(1.f / output_scale);

  if (is_broadcast_across_x)
  {
    // Select the broadcast input on the X axis
    const bool is_broadcast_input_2 = input2_win.x().step() == 0;
    Window broadcast_win = is_broadcast_input_2 ? input2_win : input1_win;
    Window non_broadcast_win = !is_broadcast_input_2 ? input2_win : input1_win;
    const ITensor *broadcast_tensor = is_broadcast_input_2 ? in2 : in1;
    const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

    const UniformQuantizationInfo broadcast_qinfo =
        broadcast_tensor->info()->quantization_info().uniform();
    const UniformQuantizationInfo non_broadcast_qinfo =
        non_broadcast_tensor->info()->quantization_info().uniform();

    const int32x4_t voffset_non_broadcast = vdupq_n_s32(non_broadcast_qinfo.offset);
    const float32x4_t vscale_non_broadcast = vdupq_n_f32(non_broadcast_qinfo.scale);

    // Clear X Dimension on execution window as we handle manually
    non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator broadcast_input(broadcast_tensor, broadcast_win);
    Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
    Iterator output(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &) {
          const auto non_broadcast_input_ptr =
              reinterpret_cast<const uint8_t *>(non_broadcast_input.ptr());
          const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

          const uint8_t broadcast_value = *reinterpret_cast<const uint8_t *>(broadcast_input.ptr());
          const float32x4x4_t broadcast_vector =
              dup_quantized(broadcast_value, broadcast_qinfo.offset, broadcast_qinfo.scale);

          int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x,
                                    non_broadcast_input_ptr, broadcast_vector, output_ptr,
                                    voffset_non_broadcast, vscale_non_broadcast, voffseto,
                                    invvscaleo, !is_broadcast_input_2);
          for (; x < window_end_x; ++x)
          {
            const float afs =
                dequantize_qasymm8(*(non_broadcast_input_ptr + x), non_broadcast_qinfo);
            const float bfs = dequantize_qasymm8(broadcast_value, broadcast_qinfo);
            *(output_ptr + x) =
                (*scalar_func)(!is_broadcast_input_2 ? bfs : afs, !is_broadcast_input_2 ? afs : bfs,
                               out->info()->quantization_info());
          }
        },
        broadcast_input, non_broadcast_input, output);
  }
  else
  {
    // Input1 quantization info
    UniformQuantizationInfo qinfo = in1->info()->quantization_info().uniform();
    const int32x4_t voffset1 = vdupq_n_s32(qinfo.offset);
    const float32x4_t vscale1 = vdupq_n_f32(qinfo.scale);

    // Input2 quantization info
    qinfo = in2->info()->quantization_info().uniform();
    const int32x4_t voffset2 = vdupq_n_s32(qinfo.offset);
    const float32x4_t vscale2 = vdupq_n_f32(qinfo.scale);

    // Clear X Dimension on execution window as we handle manually
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const QuantizationInfo input1_qinfo = in1->info()->quantization_info();
    const QuantizationInfo input2_qinfo = in2->info()->quantization_info();

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &) {
          const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
          const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
          const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

          int x =
              (*neon_func)(window_start_x, window_end_x, window_step_x, input1_ptr, input2_ptr,
                           output_ptr, voffset1, voffset2, vscale1, vscale2, voffseto, invvscaleo);
          for (; x < window_end_x; ++x)
          {
            const float afs = dequantize_qasymm8(*(input1_ptr + x), input1_qinfo);
            const float bfs = dequantize_qasymm8(*(input2_ptr + x), input2_qinfo);
            *(output_ptr + x) = (*scalar_func)(afs, bfs, out->info()->quantization_info());
          }
        },
        input1, input2, output);
  }
}

void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    float (*scalar_func)(const float &, const float &),
                    int (*broadcast_func)(int, int, int, const float *, const float &, float *,
                                          const bool),
                    int (*neon_func)(int, int, int, const float *, const float *, float *))
{
  elementwise_op_templ<float, float, float32x4_t>(in1, in2, out, window, scalar_func,
                                                  broadcast_func, neon_func);
}

void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    uint8_t (*scalar_func)(const uint8_t &, const uint8_t &),
                    int (*broadcast_func)(int, int, int, const uint8_t *, const uint8_t &,
                                          uint8_t *, const bool),
                    int (*neon_func)(int, int, int, const uint8_t *, const uint8_t *, uint8_t *))
{
  elementwise_op_templ<uint8_t, uint8_t, uint8x16_t>(in1, in2, out, window, scalar_func,
                                                     broadcast_func, neon_func);
}
} // namespace arm_compute
