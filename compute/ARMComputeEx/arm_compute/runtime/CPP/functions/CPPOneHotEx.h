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

#ifndef __ARM_COMPUTE_CPPONEHOT_EX_H__
#define __ARM_COMPUTE_CPPONEHOT_EX_H__

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref CPPOneHot */
class CPPOneHotEx : public ICPPSimpleFunction
{
public:
  /** Configure the one_hot function
   *
   * @param[in]  indices     A tensor for indices. Data types supported: S32
   * @param[in]  depth       A tensor for depth. Data types supported: S32
   * @param[in]  on_value    A tensor for on_value. Data types supported: F32
   * @param[in]  off_value   A tensor for off_value. Data types supported: F32
   * @param[out] output      A tensor for computed value of one hot operator
   * @param[in]  axis        An int value for axis
   */
  void configure(const ITensor *indices, const ITensor *depth, const ITensor *on_value,
                 const ITensor *off_value, ITensor *output, const int axis);
};
}
#endif /* __ARM_COMPUTE_CPPONEHOT_EX_H__ */
