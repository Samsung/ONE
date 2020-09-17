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

    execute_window_loop(win,
                        [&](const Coordinates &) {
                          auto output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
                          const auto non_broadcast_input_ptr =
                              reinterpret_cast<const InputScalarType *>(non_broadcast_input.ptr());
                          const InputScalarType broadcast_value =
                              *reinterpret_cast<const InputScalarType *>(broadcast_input.ptr());

                          int x = (*broadcast_func)(window_start_x, window_end_x, window_step_x,
                                                    non_broadcast_input_ptr, broadcast_value,
                                                    output_ptr, !is_broadcast_input_2);
                          for (; x < window_end_x; ++x)
                          {
                            const auto a = *(non_broadcast_input_ptr + x);
                            *(output_ptr + x) =
                                (*scalar_func)(!is_broadcast_input_2 ? broadcast_value : a,
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

    execute_window_loop(win,
                        [&](const Coordinates &) {
                          auto output_ptr = reinterpret_cast<OutputScalarType *>(output.ptr());
                          const auto input1_ptr =
                              reinterpret_cast<const InputScalarType *>(input1.ptr());
                          const auto input2_ptr =
                              reinterpret_cast<const InputScalarType *>(input2.ptr());

                          int x = (*neon_func)(window_start_x, window_end_x, window_step_x,
                                               input1_ptr, input2_ptr, output_ptr);
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
