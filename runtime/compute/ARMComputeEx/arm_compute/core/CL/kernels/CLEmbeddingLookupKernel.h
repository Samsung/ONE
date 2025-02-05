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
 * @file      CLEmbeddingLookupKernel.h
 * @ingroup   COM_AI_RUNTIME
 * @brief     This file defines CLEmbeddingLookupKernel class
 */

#ifndef __ARM_COMPUTE_CLEMBEDDINGLOOKUPKERNEL_H__
#define __ARM_COMPUTE_CLEMBEDDINGLOOKUPKERNEL_H__

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to perform EmbeddingLookup operation with opencl kernel
 */
class CLEmbeddingLookupKernel : public ICLKernel
{
public:
  /**
   * @brief Construct a CLEmbeddingLookupKernel object
   * */
  CLEmbeddingLookupKernel();

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * */
  CLEmbeddingLookupKernel(const CLEmbeddingLookupKernel &) = delete;

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * */
  CLEmbeddingLookupKernel &operator=(const CLEmbeddingLookupKernel &) = delete;

  /**
   * @brief Construct a CLEmbeddingLookupKernel object by using default move constructor
   * @param[in] CLEmbeddingLookupKernel object to move
   * */
  CLEmbeddingLookupKernel(CLEmbeddingLookupKernel &&) = default;

  /**
   * @brief Move assignment operator
   * @param[in] CLEmbeddingLookupKernel object to move
   * */
  CLEmbeddingLookupKernel &operator=(CLEmbeddingLookupKernel &&) = default;

  /**
   * @brief Destruct this object
   * */
  ~CLEmbeddingLookupKernel() = default;

  /**
   * @brief Set the input and output of the kernel
   * @param[in]  input          Source tensor.
   *                            Data type supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[out] output         Destination tensor. Data type supported: Same as @p input
   * @param[in]  lookups        Lookups are 1D tensor that values are indices into the first
   *                            dimension of input.
   *                            Data types supported: S32.
   * @return N/A
   */
  void configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *lookups);

  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   *        CLEmbeddingLookupKernel
   * @param[in]  input          The input tensor info.
   *                            Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[in]  output         The output tensor info, Data types supported: same as @p input1.
   * @param[in]  lookups        Lookups info. Data types supported: S32.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *lookups);

  /**
   * @brief Enqueue the OpenCL kernel to process the given window on the passed OpenCL command
   *        queue.
   * @note  The queue is *not* flushed by this method, and therefore the kernel will not have
   *        been executed by the time this method returns.
   * @param[in]     window  Region on which to execute the kernel. (Must be a valid region of
   *                        the window returned by window()).
   * @param[in,out] queue   Command queue on which to enqueue the kernel.@return N/A
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;   /** Source tensor */
  ICLTensor *_output;        /** Destination tensor */
  const ICLTensor *_lookups; /** Lookups tensor */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLEMBEDDINGLOOKUPKERNEL_H__ */
