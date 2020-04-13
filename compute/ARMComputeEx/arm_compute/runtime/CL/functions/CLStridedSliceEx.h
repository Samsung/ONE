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
 * Copyright (c) 2017 ARM Limited.
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

/**
 * @file CLStridedSlice.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::CLStridedSlice and arm_compute::CLStridedSliceCPU class
 */

#ifndef __ARM_COMPUTE_CLSTRIDEDSLICEEX_H__
#define __ARM_COMPUTE_CLSTRIDEDSLICEEX_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to run @ref CLStridedSliceKernel
 */
class CLStridedSliceEx : public ICLSimpleFunction
{
public:
  /**
   * @brief Initialise the kernel's inputs and outputs
   * @param[in]  input   Tensor input. Data type supported:
   *                     U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[out] output  Output tensor. Data type supported: Same as @p input
   * @param[in]  beginData 'begin' vector of strided slice operation
   * @param[in]  endData   'end' vector of strided slice operation
   * @param[in]  stridesData 'strides' vector of strided slice operation
   * @param[in]  beginMask  If the ith bit is set, begin[i] is ignored
   * @param[in]  endMask    If the ith bit is set, end[i] is ignored
   * @param[in]  shrinkAxisMask  If the ith bit is set, the ith specification shrinks the
   *                             dimensionality by 1, taking on the value at index begin[i]
   * @return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output, ICLTensor *beginData,
                 ICLTensor *endData, ICLTensor *stridesData, int32_t beginMask, int32_t endMask,
                 int32_t shrinkAxisMask);
};
}
#endif /*__ARM_COMPUTE_CLSTRIDEDSLICEEX_H__ */
