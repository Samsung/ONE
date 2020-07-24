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
 * @file CLArgOperation.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::CLArgOperation class
 */

#ifndef __ARM_COMPUTE_CLARGOPERATION_H__
#define __ARM_COMPUTE_CLARGOPERATION_H__

#include "arm_compute/core/CL/kernels/CLArgOperationKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/core/TypesEx.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to execute CLArgOperation operation
 */
class CLArgOperation : public IFunction
{
public:
  /**
   * @brief Construct a new CLArgOperation object
   */
  CLArgOperation();

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLArgOperation(const CLArgOperation &) = delete;

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLArgOperation &operator=(const CLArgOperation &) = delete;

  /**
   * @brief Construct a new CLArgOperation object by using copy constructor
   * @param[in] CLArgOperation object to move
   */
  CLArgOperation(CLArgOperation &&) = default;

  /**
   * @brief Assign a CLArgOperation object.
   * @param[in] CLArgOperation object to assign. This object will be moved.
   */
  CLArgOperation &operator=(CLArgOperation &&) = default;

  /**
   * @brief Initialise the kernel's inputs and outputs.
   * @param[in]  input     Input tensor. Data types supported: U8/QASYMM8/S32/F32.
   * @param[out] output    The result of arg operation. Data types supported: S32.
   * @param[in]  axis      Axis along which to reduce. It must be sorted and no duplicates.
   * @param[in]  op        Arg operation to perform.
   * @return N/A
   */
  void configure(ICLTensor *input, ICLTensor *output, std::vector<uint32_t> axis, ArgOperation op);

  /**
   * @brief Static function to check if given info will lead to a valid configuration
   * @param[in]  input     Input tensor. Data types supported: U8/QASYMM8/S32/F32.
   * @param[in]  axis      Axis along which to reduce. It must be sorted and no duplicates.
   * @param[out] output    The result of arg operation. Data types supported: S32.
   * @param[in]  op        Arg operation to perform.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const std::vector<uint32_t> &axis,
                         const ITensorInfo *output, ArgOperation op);
  /**
   * @brief Run the OpenCL kernel for this operation
   * @return N/A
   */
  void run() override;

private:
  ICLTensor *_input{nullptr};
  ICLTensor *_output{nullptr};
  std::vector<uint32_t> _axis{};
  ArgOperation _arg_op{ArgOperation::MAX};

  std::unique_ptr<CLTensor[]> _interm_tensors{nullptr};
  std::unique_ptr<CLArgOperationKernel[]> _argop_kernels{nullptr};
  size_t _num_of_kernels{0};
};
}
#endif /*__ARM_COMPUTE_CLARGOPERATION_H__ */
