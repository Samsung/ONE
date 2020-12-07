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
 * @file      CLHashtableLookupKernel.h
 * @ingroup   COM_AI_RUNTIME
 * @brief     This file defines CLHashtableLookupKernel class
 */

#ifndef __ARM_COMPUTE_CLHASHTABLELOOKUPKERNEL_H__
#define __ARM_COMPUTE_CLHASHTABLELOOKUPKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to perform HashtableLookup operation with opencl kernel
 */
class CLHashtableLookupKernel : public ICLKernel
{
public:
  /**
   * @brief Construct a CLHashtableLookupKernel object
   * */
  CLHashtableLookupKernel();

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * */
  CLHashtableLookupKernel(const CLHashtableLookupKernel &) = delete;

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * */
  CLHashtableLookupKernel &operator=(const CLHashtableLookupKernel &) = delete;

  /**
   * @brief Construct a CLHashtableLookupKernel object by using default move constructor
   * @param[in] CLHashtableLookupKernel object to move
   * */
  CLHashtableLookupKernel(CLHashtableLookupKernel &&) = default;

  /**
   * @brief Move assignment operator
   * @param[in] CLHashtableLookupKernel object to move
   * */
  CLHashtableLookupKernel &operator=(CLHashtableLookupKernel &&) = default;

  /**
   * @brief Destruct this object
   * */
  ~CLHashtableLookupKernel() = default;

  /**
   * @brief Set the input and output of the kernel
   * @param[in]  lookups  Lookups 1D tensor that values are indices into the first dimension of
   *                      input.
   * @param[in]  keys     Keys 1D tensor. keys and input pair represent a map.
   *                      Data types supported: S32
   * @param[in]  input    Source tensor.
   *                      Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[out] output   Destination tensor. Data types and data layouts supported: Same as @p
   *                      input.
   * @param[out] hits     Hits 1D tensor. A boolean tensor that indicates whether the lookup hits
   *                      (True) or not (False). Data types supported: U8/QASYMM8
   * @return N/A
   */
  void configure(const ICLTensor *lookups, const ICLTensor *keys, const ICLTensor *input,
                 ICLTensor *output, ICLTensor *hits);

  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   *        CLHashtableLookupKernel
   * @param[in]  lookups  The lookups tensor info. Data types supported: S32.
   * @param[in]  keys     The keys tensor info. keys and input pair represent a map.
   *                      Data types supported: S32
   * @param[in]  input    The input tensor info.
   *                      Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
   * @param[out] output   The output tensor. Data types and data layouts supported: Same as @p
   *                      input.
   * @param[out] hits     The hits tensor info. A boolean tensor that indicates whether the lookup
   *                      hits
   *                      (True) or not (False). Data types supported: U8/QASYMM8
   * @return a status
   */
  static Status validate(const ITensorInfo *lookups, const ITensorInfo *keys,
                         const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *hits);

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
  const ICLTensor *_lookups{nullptr};                 /** Lookups tensor */
  const ICLTensor *_keys{nullptr};                    /** Keys tensor */
  const ICLTensor *_input{nullptr};                   /** Source tensor */
  ICLTensor *_output{nullptr};                        /** Destination tensor */
  ICLTensor *_hits{nullptr};                          /** Hits tensor */
  std::unique_ptr<CLTensor> _lookup_indices{nullptr}; /** Lookup indices tensor */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLHASHTABLELOOKUPKERNEL_H__ */
