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

/**
 * @file CLPixelWiseDivision.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::CLPixelWiseDivision class
 */
#ifndef __ARM_COMPUTE_CLPIXELWISEDIVISION_H__
#define __ARM_COMPUTE_CLPIXELWISEDIVISION_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to run @ref CLPixelWiseDivisionKernel.
 */
class CLPixelWiseDivision : public ICLSimpleFunction
{
public:
  /**
   * @brief Initialise the kernel's inputs, output and convertion policy.
   * @param[in, out] input1          An input tensor. Data types supported: U8/S16/F16/F32
   *                                 The input tensor is [in, out] because its TensorInfo might be
   * modified inside the kernel in case of broadcasting of dimension 0.
   * @param[in, out] input2          An input tensor. Data types supported: same as @p input1.
   *                                 The input tensor is [in, out] because its TensorInfo might be
   * modified inside the kernel in case of broadcasting of dimension 0.
   * @param[out]     output          The output tensor, Data types supported: same as @p input1.
   * Note: U8 requires both inputs to be U8.
   * @param[in]      scale           Scale to apply after multiplication.
   *                                 Scale must be positive and its value must be either 1/255 or
   * 1/2^n where n is between 0 and 15.
   * @param[in]      overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
   * @param[in]      rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest
   * even.
   * @return N/A
   */
  void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale = 1.f,
                 ConvertPolicy overflow_policy = ConvertPolicy::WRAP,
                 RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO);

  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   * CLPixelWiseDivision
   * @param[in] input1          An input tensor info. Data types supported: U8/S16/F16/F32
   * @param[in] input2          An input tensor info. Data types supported: same as @p input1.
   * @param[in] output          The output tensor info, Data types supported: same as @p input1.
   * Note: U8 requires both inputs to be U8.
   * @param[in] scale           Scale to apply after multiplication.
   *                            Scale must be positive and its value must be either 1/255 or 1/2^n
   * where n is between 0 and 15.
   * @param[in] overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
   * @param[in] rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
   * @return a status
   */
  static Status validate(const ITensorInfo *input1, const ITensorInfo *input2,
                         const ITensorInfo *output, float scale = 1.f,
                         ConvertPolicy overflow_policy = ConvertPolicy::WRAP,
                         RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO);
};
}
#endif /*__ARM_COMPUTE_CLPIXELWISEDIVISION_H__ */
