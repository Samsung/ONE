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
 * @file CLArgOperationKernel.h
 * @brief This file defines CLArgOperationKernel
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_CLARGOPERATIONKERNEL_H__
#define __ARM_COMPUTE_CLARGOPERATIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/TypesEx.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to define interface for the argop kernel.
 */
class CLArgOperationKernel : public ICLKernel
{
public:
  /**
   * @brief Default constructor.
   */
  CLArgOperationKernel();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLArgOperationKernel to be copied
   */
  CLArgOperationKernel(const CLArgOperationKernel &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLArgOperationKernel to be copied
   * @return Reference of this instance
   */
  CLArgOperationKernel &operator=(const CLArgOperationKernel &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLArgOperationKernel to be moved
   */
  CLArgOperationKernel(CLArgOperationKernel &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLArgOperationKernel to be moved
   * @return Reference of this instance
   */
  CLArgOperationKernel &operator=(CLArgOperationKernel &&) = default;
  /**
   * @brief Initialise the kernel's input, output and border mode.
   * @param[in]  input          An input tensor. Data types supported: U8/QASYMM8/S32/F32.
   * @param[out] output         The output tensor, Data types supported: S32.
   * @param[in]  axis           Axis along which to reduce. It must be sorted and no duplicates.
   * @param[in]  op             Arg operation to perform.
   * return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output, const uint32_t axis, ArgOperation op);
  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   * CLArgOperationKernel
   * @param[in] input           An input tensor info. Data types supported: U8/QASYMM8/S32/F32.
   * @param[in] output          The output tensor info, Data types supported: S32.
   * @param[in] axis            Axis along which to reduce. It must be sorted and no duplicates.
   * @param[in] op              Arg operation to perform.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output, const uint32_t axis,
                         ArgOperation op);

  /*
   * @brief Run CLArgOperationKernel op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;
  ICLTensor *_output;
  uint32_t _axis;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLARGOPERATIONKERNEL_H__ */
