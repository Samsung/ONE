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

#ifndef __ARM_COMPUTE_UTILSEX_H__
#define __ARM_COMPUTE_UTILSEX_H__

#include <utility>

#include "arm_compute/core/Types.h"

namespace arm_compute
{

/** Returns expected width and height of the transpose convolution's output tensor.
 *
 * @note This function was copied in order to fix a bug computing to wrong output dimensions.
 *
 * @param[in] in_width      Width of input tensor (Number of columns)
 * @param[in] in_height     Height of input tensor (Number of rows)
 * @param[in] kernel_width  Kernel width.
 * @param[in] kernel_height Kernel height.
 * @param[in] info          padding and stride info.
 * @param[in] invalid_right The number of zeros added to right edge of the output.
 * @param[in] invalid_top   The number of zeros added to bottom edge of the output.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
const std::pair<unsigned int, unsigned int>
transposeconv_output_dimensions(unsigned int in_width, unsigned int in_height,
                                unsigned int kernel_width, unsigned int kernel_height,
                                const PadStrideInfo &info, unsigned int invalid_right,
                                unsigned int invalid_top);
} // namespace arm_compute
#endif /*__ARM_COMPUTE_UTILSEX_H__ */
