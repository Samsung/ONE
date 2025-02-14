/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "arm_compute/core/UtilsEx.h"
#include "arm_compute/core/Error.h"

using namespace arm_compute;

const std::pair<unsigned int, unsigned int>
arm_compute::transposeconv_output_dimensions(unsigned int in_width, unsigned int in_height,
                                             unsigned int kernel_width, unsigned int kernel_height,
                                             const PadStrideInfo &info, unsigned int invalid_right,
                                             unsigned int invalid_bottom)
{
  const auto [stride_x, stride_y] = info.stride();
  const unsigned int padx = info.pad_left() + info.pad_right();
  const unsigned int pady = info.pad_top() + info.pad_bottom();

  ARM_COMPUTE_ERROR_ON(in_width < 1 || in_height < 1);
  ARM_COMPUTE_ERROR_ON(kernel_width <= padx);
  ARM_COMPUTE_ERROR_ON(kernel_height <= pady);

  // Find the transpose conv out dimensions
  // transpose conv out:
  //    tconv_out + pad = 1 + (in - 1) * stride + invalid
  //    tconv_out = 1 + (in - 1) * stride + invalid - pad
  const int w = stride_x * (in_width - 1) + kernel_width - padx + invalid_right;
  const int h = stride_y * (in_height - 1) + kernel_height - pady + invalid_bottom;

  return std::make_pair<unsigned int, unsigned int>(w, h);
}
