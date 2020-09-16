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

#ifndef __ARM_COMPUTE_NEELEMENTWISEOPERATIONFUNCS_H__
#define __ARM_COMPUTE_NEELEMENTWISEOPERATIONFUNCS_H__

#include <arm_neon.h>

namespace arm_compute
{
class ITensor;
class Window;
class QuantizationInfo;
} // namespace arm_compute

namespace arm_compute
{

void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    float (*scalar_func)(const float &, const float &),
                    int (*broadcast_func)(int, int, int, const float *, const float &, float *,
                                          const bool),
                    int (*neon_func)(int, int, int, const float *, const float *, float *));

void elementwise_op(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window,
                    uint8_t (*scalar_func)(const uint8_t &, const uint8_t &),
                    int (*broadcast_func)(int, int, int, const uint8_t *, const uint8_t &,
                                          uint8_t *, const bool),
                    int (*neon_func)(int, int, int, const uint8_t *, const uint8_t *, uint8_t *));
} // namespace arm_compute
#endif // __ARM_COMPUTE_NEELEMENTWISEOPERATIONFUNCS_H__
