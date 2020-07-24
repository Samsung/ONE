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
 * @file      CLCastKernel.h
 * @ingroup   COM_AI_RUNTIME
 * @brief     This file defines CLCastKernel class
 */

#ifndef __ARM_COMPUTE_CLCASTKERNEL_H__
#define __ARM_COMPUTE_CLCASTKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/TypesEx.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to define OpenCL kernel for cast operation
 */
class CLCastKernel : public ICLKernel
{
public:
  /**
   * @brief Construct CLCastKernel object
   */
  CLCastKernel();

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLCastKernel(const CLCastKernel &) = delete;

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLCastKernel &operator=(const CLCastKernel &) = delete;

  /**
   * @brief Construct CLCastKernel object using default move constructor
   * @param[in] CLCastKernel object to move
   */
  CLCastKernel(CLCastKernel &&) = default;

  /**
   * @brief Allow instances of this class to be moved
   * @param[in] CLCastKernel object to move
   */
  CLCastKernel &operator=(CLCastKernel &&) = default;

  /**
   * @brief Destruct this CLCastKernel object
   */
  ~CLCastKernel() = default;

  /**
   * @brief Initialise the kernel's input and output.
   * @param[in]  input  Input tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   * @param[in]  output Output tensor. Data types supported: U8/QASYMM8/S16/S32/F16/F32.
   * @param[in]  input_subtype  Sub data type of input.
   * @return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output, SubDataType input_subtype);

  /**
   * @brief Enqueue the OpenCL kernel to process the given window on the passed OpenCL command
   *        queue.
   * @note  The queue is *not* flushed by this method, and therefore the kernel will not have
   *        been executed by the time this method returns.
   * @param[in] window      Region on which to execute the kernel. (Must be a valid region of
   *                        the window returned by window()).
   * @param[in,out] queue   Command queue on which to enqueue the kernel.@return N/A
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input; /**< Source tensor */
  ICLTensor *_output;      /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCASTKERNEL_H__ */
