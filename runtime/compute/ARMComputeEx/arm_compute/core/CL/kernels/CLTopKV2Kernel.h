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
 * @file CLTopKV2Kernel.h
 * @brief This file defines classes for TopKV2Kernel
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_CLTOPKV2KERNEL_H__
#define __ARM_COMPUTE_CLTOPKV2KERNEL_H__

#include "src/core/CL/ICLKernel.h"

// these parameters can be changed
#define _ITEMS 16                          // number of items in a group
#define _GROUPS 4                          // the number of virtual processors is _ITEMS * _GROUPS
#define _HISTOSPLIT (_ITEMS * _GROUPS / 2) // number of splits of the histogram
#define PERMUT                             // store the final permutation
////////////////////////////////////////////////////////

// Disable GPU implementation
// TODO Enable GPU implementation with verification, or remove code
//      Invalid result on GPU
#if 0
namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to define CLTopKV2Single
 */
class CLTopKV2Single : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLTopKV2Single();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Single to be copied
   */
  CLTopKV2Single(const CLTopKV2Single &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Single to be copied
   * @return Reference of this instance
   */
  CLTopKV2Single &operator=(const CLTopKV2Single &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Single to be moved
   */
  CLTopKV2Single(CLTopKV2Single &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Single to be moved
   * @return Reference of this instance
   */
  CLTopKV2Single &operator=(CLTopKV2Single &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[in] input An input tensor
   * @param[in] topk_values Values of the top k predictions
   * @param[in] topk_indices Indices of the top k predictions
   * @param[in] indices Indices
   * @param[in] temp_stack Temp stack
   * @param[in] k K of the top k predictions
   * @param[in] n Number times to quick-sort
   * return N/A
   */
  void configure(ICLTensor *input, ICLTensor *topk_values, ICLTensor *topk_indices,
                 cl::Buffer *indices, cl::Buffer *temp_stack, int k, int n);

  /*
   * @brief Run CLTopKV2Single op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_input;
  ICLTensor *_topk_values;
  ICLTensor *_topk_indices;
};

/**
 * @brief Class to define CLTopKV2Init
 */
class CLTopKV2Init : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLTopKV2Init();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Init to be copied
   */
  CLTopKV2Init(const CLTopKV2Init &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Init to be copied
   * @return Reference of this instance
   */
  CLTopKV2Init &operator=(const CLTopKV2Init &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Init to be moved
   */
  CLTopKV2Init(CLTopKV2Init &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Init to be moved
   * @return Reference of this instance
   */
  CLTopKV2Init &operator=(CLTopKV2Init &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[in] input An input tensor
   * @param[in] in_key_buf Buffer of input key
   * @param[in] in_ind_buf Buffer of input index
   * @param[in] n Number times to quick-sort
   * return N/A
   */
  void configure(ICLTensor *input, cl::Buffer *in_key_buf, cl::Buffer *in_ind_buf, int n);

  /*
   * @brief Run CLTopKV2Init op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_input;
};

/**
 * @brief Class to define CLRadixSortHistogram
 */
class CLRadixSortHistogram : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLRadixSortHistogram();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortHistogram to be copied
   */
  CLRadixSortHistogram(const CLRadixSortHistogram &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortHistogram to be copied
   * @return Reference of this instance
   */
  CLRadixSortHistogram &operator=(const CLRadixSortHistogram &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortHistogram to be moved
   */
  CLRadixSortHistogram(CLRadixSortHistogram &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortHistogram to be moved
   * @return Reference of this instance
   */
  CLRadixSortHistogram &operator=(CLRadixSortHistogram &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] hist_buf Buffer of histogram
   * @param[in] bits Number of bits to be used for radix sort
   * @param[in] n Integer number size to sort
   * return N/A
   */
  void configure(cl::Buffer *hist_buf, int bits, int n);

  /**
   * @brief Set pass
   * @param[in] pass Passes made of in radix sort algorithm
   * @param[in] in_key_buf Buffer of input key
   * return N/A
   */
  void setPass(int pass, cl::Buffer *in_key_buf)
  {
    _pass = pass;
    _in_key_buf = in_key_buf;
  }

  /*
   * @brief Run CLRadixSortHistogram op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  int _pass;
  cl::Buffer *_in_key_buf;
};

/**
 * @brief Class to define CLRadixSortScanHistogram
 */
class CLRadixSortScanHistogram : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLRadixSortScanHistogram();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortScanHistogram to be copied
   */
  CLRadixSortScanHistogram(const CLRadixSortScanHistogram &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortScanHistogram to be copied
   * @return Reference of this instance
   */
  CLRadixSortScanHistogram &operator=(const CLRadixSortScanHistogram &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortScanHistogram to be moved
   */
  CLRadixSortScanHistogram(CLRadixSortScanHistogram &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortScanHistogram to be moved
   * @return Reference of this instance
   */
  CLRadixSortScanHistogram &operator=(CLRadixSortScanHistogram &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] hist_buf Buffer of histogram
   * @param[out] glob_sum_buf Buffer of global sum
   * @param[in] bits Number of bits to be used for radix sort
   * return N/A
   */
  void configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits);

  /*
   * @brief Run CLRadixSortScanHistogram op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;
};

/**
 * @brief Class to define CLRadixSortGlobalScanHistogram
 */
class CLRadixSortGlobalScanHistogram : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLRadixSortGlobalScanHistogram();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortGlobalScanHistogram to be copied
   */
  CLRadixSortGlobalScanHistogram(const CLRadixSortGlobalScanHistogram &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortGlobalScanHistogram to be copied
   * @return Reference of this instance
   */
  CLRadixSortGlobalScanHistogram &operator=(const CLRadixSortGlobalScanHistogram &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortGlobalScanHistogram to be moved
   */
  CLRadixSortGlobalScanHistogram(CLRadixSortGlobalScanHistogram &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortGlobalScanHistogram to be moved
   * @return Reference of this instance
   */
  CLRadixSortGlobalScanHistogram &operator=(CLRadixSortGlobalScanHistogram &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] glob_sum_buf Buffer of global sum
   * @param[out] temp_buf Temp buffer to be used while RadixSortGlobalScanHistogram
   * @param[in] bits Number of bits to be used for radix sort
   * return N/A
   */
  void configure(cl::Buffer *glob_sum_buf, cl::Buffer *temp_buf, int bits);

  /*
   * @brief Run CLRadixSortGlobalScanHistogram op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;
};

/**
 * @brief Class to define CLRadixSortPasteHistogram
 */
class CLRadixSortPasteHistogram : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLRadixSortPasteHistogram();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortPasteHistogram to be copied
   */
  CLRadixSortPasteHistogram(const CLRadixSortPasteHistogram &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortPasteHistogram to be copied
   * @return Reference of this instance
   */
  CLRadixSortPasteHistogram &operator=(const CLRadixSortPasteHistogram &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortPasteHistogram to be moved
   */
  CLRadixSortPasteHistogram(CLRadixSortPasteHistogram &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortPasteHistogram to be moved
   * @return Reference of this instance
   */
  CLRadixSortPasteHistogram &operator=(CLRadixSortPasteHistogram &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] hist_buf Buffer of histogram
   * @param[out] glob_sum_buf Buffer of global sum
   * @param[in] bits Number of bits to be used for radix sort
   * return N/A
   */
  void configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits);

  /*
   * @brief Run CLRadixSortPasteHistogram op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;
};

/**
 * @brief Class to define CLRadixSortReorder
 */
class CLRadixSortReorder : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLRadixSortReorder();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortReorder to be copied
   */
  CLRadixSortReorder(const CLRadixSortReorder &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLRadixSortReorder to be copied
   * @return Reference of this instance
   */
  CLRadixSortReorder &operator=(const CLRadixSortReorder &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortReorder to be moved
   */
  CLRadixSortReorder(CLRadixSortReorder &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLRadixSortReorder to be moved
   * @return Reference of this instance
   */
  CLRadixSortReorder &operator=(CLRadixSortReorder &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] hist_buf Buffer of histogram
   * @param[in] bits Number of bits to be used for radix sort
   * @param[in] n Integer number size to sort
   * return N/A
   */
  void configure(cl::Buffer *hist_buf, int bits, int n);

  /**
   * @brief Set pass
   * @param[in] pass Passes made of in radix sort algorithm
   * @param[in] in_key_buf Buffer of input key
   * @param[out] out_key_buf Buffer of output key
   * @param[in] in_ind_buf Buffer of input index
   * @param[out] out_ind_buf Buffer of output index
   * return N/A
   */
  void setPass(int pass, cl::Buffer *in_key_buf, cl::Buffer *out_key_buf, cl::Buffer *in_ind_buf,
               cl::Buffer *out_ind_buf)
  {
    _pass = pass;
    _in_key_buf = in_key_buf;
    _out_key_buf = out_key_buf;
    _in_ind_buf = in_ind_buf;
    _out_ind_buf = out_ind_buf;
  }
  /*
   * @brief Run CLRadixSortReorder op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  int _pass;
  cl::Buffer *_in_key_buf;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_in_ind_buf;
  cl::Buffer *_out_ind_buf;
};

/**
 * @brief Class to define CLTopKV2FindFirstNegative
 */
class CLTopKV2FindFirstNegative : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLTopKV2FindFirstNegative();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2FindFirstNegative to be copied
   */
  CLTopKV2FindFirstNegative(const CLTopKV2FindFirstNegative &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2FindFirstNegative to be copied
   * @return Reference of this instance
   */
  CLTopKV2FindFirstNegative &operator=(const CLTopKV2FindFirstNegative &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2FindFirstNegative to be moved
   */
  CLTopKV2FindFirstNegative(CLTopKV2FindFirstNegative &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2FindFirstNegative to be moved
   * @return Reference of this instance
   */
  CLTopKV2FindFirstNegative &operator=(CLTopKV2FindFirstNegative &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] first_negative_idx_buf Buffer of the first negative index
   * @param[in] n Number times to find
   * return N/A
   */
  void configure(cl::Buffer *first_negative_idx_buf, int n);

  /**
   * @brief Set output buffer
   * @param[out] out_key_buf Buffer of output key
   * return N/A
   */
  void setOutputBuffer(cl::Buffer *out_key_buf) { _out_key_buf = out_key_buf; }

  /*
   * @brief Run CLTopKV2FindFirstNegative op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  cl::Buffer *_out_key_buf;
};

/**
 * @brief Class to define CLTopKV2ReorderNegatives
 */
class CLTopKV2ReorderNegatives : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLTopKV2ReorderNegatives();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2ReorderNegatives to be copied
   */
  CLTopKV2ReorderNegatives(const CLTopKV2ReorderNegatives &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2ReorderNegatives to be copied
   * @return Reference of this instance
   */
  CLTopKV2ReorderNegatives &operator=(const CLTopKV2ReorderNegatives &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2ReorderNegatives to be moved
   */
  CLTopKV2ReorderNegatives(CLTopKV2ReorderNegatives &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2ReorderNegatives to be moved
   * @return Reference of this instance
   */
  CLTopKV2ReorderNegatives &operator=(CLTopKV2ReorderNegatives &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] first_negative_idx_buf Buffer of the first negative index
   * @param[in] n Number times to find
   * return N/A
   */
  void configure(cl::Buffer *first_negative_idx_buf, int n);

  /**
   * @brief Set buffers
   * @param[in] in_key_buf Buffer of input key
   * @param[out] out_key_buf Buffer of output key
   * @param[in] in_ind_buf Buffer of input index
   * @param[out] out_ind_buf Buffer of output index
   * return N/A
   */
  void setBuffers(cl::Buffer *in_key_buf, cl::Buffer *out_key_buf, cl::Buffer *in_ind_buf,
                  cl::Buffer *out_ind_buf)
  {
    _in_key_buf = in_key_buf;
    _out_key_buf = out_key_buf;
    _in_ind_buf = in_ind_buf;
    _out_ind_buf = out_ind_buf;
  }

  /*
   * @brief Run CLTopKV2ReorderNegatives op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  cl::Buffer *_in_key_buf;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_in_ind_buf;
  cl::Buffer *_out_ind_buf;
};

/**
 * @brief Class to define CLTopKV2Store
 */
class CLTopKV2Store : public ICLKernel
{
public:
  /**
   * @brief Constructor
   */
  CLTopKV2Store();
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Store to be copied
   */
  CLTopKV2Store(const CLTopKV2Store &) = delete;
  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers).
   * @param [in] copiedInstance Const reference of CLTopKV2Store to be copied
   * @return Reference of this instance
   */
  CLTopKV2Store &operator=(const CLTopKV2Store &) = delete;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Store to be moved
   */
  CLTopKV2Store(CLTopKV2Store &&) = default;
  /**
   * @brief Allow instances of this class to be moved
   * @param [in] movedInstance Rvalue reference of CLTopKV2Store to be moved
   * @return Reference of this instance
   */
  CLTopKV2Store &operator=(CLTopKV2Store &&) = default;

  /**
   * @brief Initialise kernel with params
   * @param[out] values Values tensor to store
   * @param[out] indices Indices tensor to be used for store
   * @param[in] k K of the top k predictions
   * @param[in] n Number times to store
   * return N/A
   */
  void configure(ICLTensor *values, ICLTensor *indices, int k, int n);

  /**
   * @brief Set buffers
   * @param[out] out_key_buf Buffer of output key
   * @param[out] out_ind_buf Buffer of output index
   * return N/A
   */
  void setOutputBuffers(cl::Buffer *out_key_buf, cl::Buffer *out_ind_buf);

  /*
   * @brief Run CLTopKV2Store op
   * @param[in] window  Window to be used for in_slice
   * @param[in] queue   cl::CommandQueue
   * @return N/A
   */
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  ICLTensor *_values;
  ICLTensor *_indices;
  cl::Buffer *_out_key_buf;
  cl::Buffer *_out_ind_buf;
};

} // namespace arm_compute
#endif // Disable GPU implementation
#endif // __ARM_COMPUTE_CLTOPKV2KERNEL_H__
