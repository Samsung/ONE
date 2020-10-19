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
 * @file CLReduceOperationKernel.h
 * @brief This file defines CLReduceOperationKernel class
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_CLREDUCEOPERATIONKERNEL_H__
#define __ARM_COMPUTE_CLREDUCEOPERATIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to define interface for the reduce operation kernel
 */
class CLReduceOperationKernel : public ICLKernel
{
public:
  /**
   * @brief Default constructor
   */
  CLReduceOperationKernel();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLReduceOperationKernel(const CLReduceOperationKernel &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLReduceOperationKernel &operator=(const CLReduceOperationKernel &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   */
  CLReduceOperationKernel(CLReduceOperationKernel &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   */
  CLReduceOperationKernel &operator=(CLReduceOperationKernel &&) = default;
  /**
   * @brief Default destructor
   */
  ~CLReduceOperationKernel() = default;

  /**
   * @brief Set the input and output tensors.
   * @param[in]  input  Source tensor. Data types supported: U8/S32/F32.
   * @param[out] output Destination tensor. Data types supported: Same as @p input.
   *                    Output will have the same number of dimensions as input.
   * @param[in]  axis   Axis along which to reduce.
   * @param[in]  op     Reduce operation to perform.
   * @return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output, const uint32_t axis,
                 ReductionOperation op);

  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   *        CLReduceOperationKernel.
   * @param[in] input  Source tensor info. Data types supported: U8/S32/F32.
   * @param[in] output Destination tensor info. Data types supported: Same as @p input.
   *                   Output will have the same number of dimensions as input.
   * @param[in] axis   Axis along which to reduce.
   * @param[in] op     Reduce operation to perform.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output, const uint32_t axis,
                         ReductionOperation op);

  /*
   * @brief Run CLReduceOperationKernel op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   CLQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;
  ICLTensor *_output;
  uint32_t _axis;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREDUCEOPERATIONKERNEL_H__ */
