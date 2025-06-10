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
 * @file CLTopKV2.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::CLTopKV2 class
 */
#ifndef __ARM_COMPUTE_CLTOPK_V2_H__
#define __ARM_COMPUTE_CLTOPK_V2_H__

#include "arm_compute/core/CL/kernels/CLTopKV2Kernel.h"

#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to execute TopKV2 operation.
 */
class CLTopKV2 : public IFunction
{
public:
  /**
   * @brief Construct a new CLTopKV2 object
   */
  CLTopKV2();

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLTopKV2(const CLTopKV2 &) = delete;

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   */
  CLTopKV2 &operator=(const CLTopKV2 &) = delete;

  /**
   * @brief Construct a new CLTopKV2 object by using copy constructor
   * @param[in] CLTopKV2 object to move
   */
  CLTopKV2(CLTopKV2 &&) = default;

  /**
   * @brief Assign a CLTopKV2 object.
   * @param[in] CLTopKV2 object to assign. This object will be moved.
   */
  CLTopKV2 &operator=(CLTopKV2 &&) = default;

  /**
   * @brief Initialise the kernel's inputs and outputs.
   * @param[in]  input     Input image. Data types supported: U8/S16/F32.
   * @param[in]  k         The value of `k`.
   * @param[out] values    Top k values. Data types supported: S32 if input type is U8/S16, F32 if
   * input type is F32.
   * @param[out] indices   Indices related to top k values. Data types supported: S32 if input type
   * is U8/S16, F32 if input type is F32.
   * @return N/A
   */
  void configure(ICLTensor *input, int k, ICLTensor *values, ICLTensor *indices,
                 int total_bits = 32, int bits = 4);

  /**
   * @brief Run the kernels contained in the function
   * Depending on the value of the following environment variables it works differently:
   *   - If the value of environment variable "ACL_TOPKV2" == "GPU_SINGLE",
   *     quick sort on GPU is used.
   *   - If the value of environment variable "ACL_TOPKV2" == ""GPU"",
   *     radix sort on GPU is used.
   *   - For other value, TopKV2 runs on CPU
   * @return N/A
   */
  void run() override;

private:
  void run_on_cpu();
  void run_on_gpu();
  void run_on_gpu_single_quicksort();

  uint32_t _k;
  uint32_t _total_bits;
  uint32_t _bits;
  uint32_t _radix;
  uint32_t _hist_buf_size;
  uint32_t _glob_sum_buf_size;
  uint32_t _n;

  ICLTensor *_input;
  ICLTensor *_values;
  ICLTensor *_indices;

  cl::Buffer _qs_idx_buf;
  cl::Buffer _qs_temp_buf;
  cl::Buffer _hist_buf;
  cl::Buffer _glob_sum_buf;
  cl::Buffer _temp_buf;
  cl::Buffer _first_negative_idx_buf;
  cl::Buffer _in_key_buf;
  cl::Buffer _out_key_buf;
  cl::Buffer _in_ind_buf;
  cl::Buffer _out_ind_buf;

  cl::Buffer *_p_in_key_buf;
  cl::Buffer *_p_out_key_buf;
  cl::Buffer *_p_in_ind_buf;
  cl::Buffer *_p_out_ind_buf;
// Disable GPU implementation
// TODO Enable GPU implementation with verification, or remove code
//      Invalid result on GPU
#if 0
  CLTopKV2Single _qs_kernel;
  CLTopKV2Init _init_kernel;
  CLRadixSortHistogram _hist_kernel;
  CLRadixSortScanHistogram _scan_hist_kernel;
  CLRadixSortGlobalScanHistogram _glob_scan_hist_kernel;
  CLRadixSortPasteHistogram _paste_hist_kernel;
  CLRadixSortReorder _reorder_kernel;
  CLTopKV2FindFirstNegative _find_first_negative_kernel;
  CLTopKV2ReorderNegatives _reorder_negatives_kernel;
  CLTopKV2Store _store_kernel;
#endif
};
} // namespace arm_compute
#endif // __ARM_COMPUTE_CLTOPK_V2_H__
