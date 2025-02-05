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
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef __ARM_COMPUTE_CLONEHOTKERNEL_H__
#define __ARM_COMPUTE_CLONEHOTKERNEL_H__
#include "src/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"
namespace arm_compute
{
class ICLTensor;
/** Interface for the kernel to perform one-hot encoding*/
class CLOneHotKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLOneHotKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLOneHotKernel(const CLOneHotKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLOneHotKernel &operator=(const CLOneHotKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLOneHotKernel(CLOneHotKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLOneHotKernel &operator=(CLOneHotKernel &&) = default;
  /** Default destructor */
  ~CLOneHotKernel() = default;
  /** Initialise the kernel's inputs and output
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[in]  off_value Off value tensor. Supported tensor rank: only 1. Data type supported:
   * Same as @p on_value
   * @param[out] output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  depth     The depth of the one hot dimension.
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * value must be in range [-indices.rank , indices.rank)
   */
  void configure(const ICLTensor *indices, const ICLTensor *on_value, const ICLTensor *off_value,
                 ICLTensor *output, int depth, int axis = -1);
  /** Initialise the kernel's inputs and output already initialized to off_value
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[out] output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  depth     The depth of the one hot dimension.
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * value must be in range [-indices.rank , indices.rank)
   */
  void configure(const ICLTensor *indices, const ICLTensor *on_value, ICLTensor *output, int depth,
                 int axis = -1);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLOneHotKernel
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[in]  off_value Off value tensor. Supported tensor rank: only 1. Data type supported:
   * Same as @p on_value
   * @param[in]  output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  depth     The depth of the one hot dimension.
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * value must be in range [-indices.rank , indices.rank)
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *indices, const ITensorInfo *on_value,
                         const ITensorInfo *off_value, const ITensorInfo *output, int depth,
                         int axis = -1);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLOneHotKernel without off_value
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[in]  output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  depth     The depth of the one hot dimension.
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * value must be in range [-indices.rank , indices.rank)
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *indices, const ITensorInfo *on_value,
                         const ITensorInfo *output, int depth, int axis = -1);
  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  /** Initialise the kernel's inputs and outputs internally
   *
   * @param[in]  indices   Indices tensor. Supported tensor rank: up to 3. Must be one of the
   * following types: U32/S32
   * @param[in]  on_value  On value tensor. Supported tensor rank: only 1. Data type supported:
   * U8/S8/U16/S16/F16/U32/S32/F32
   * @param[out] output    Destination tensor. Data type supported: Same as @p on_value
   * @param[in]  depth     The depth of the one hot dimension.
   * @param[in]  axis      (Optional) The axis to fill. Negative values wrap around. Defaults to -1.
   * value must be in range [-indices.rank , indices.rank)
   */
  void configure_common(const ICLTensor *indices, const ICLTensor *on_value, ICLTensor *output,
                        int depth, int axis);

private:
  const ICLTensor *_indices;   /**< Indices tensor */
  const ICLTensor *_on_value;  /**< On value tensor */
  const ICLTensor *_off_value; /**< Off value tensor */
  ICLTensor *_output;          /**< Destination tensor */
  bool _is_off_value_memset;   /**< Whether off_value is zero */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLONEHOTKERNEL_H__ */
