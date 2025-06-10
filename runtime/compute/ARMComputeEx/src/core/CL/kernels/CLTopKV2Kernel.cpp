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

#include "arm_compute/core/CL/kernels/CLTopKV2Kernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

// Disable GPU implementation
// TODO Enable GPU implementation with verification, or remove code
//      Invalid result on GPU
#if 0
namespace arm_compute
{
////////////////////////////////////////////////////////////////////////////////
CLTopKV2Single::CLTopKV2Single() : _input(nullptr), _topk_values(nullptr), _topk_indices(nullptr) {}

void CLTopKV2Single::configure(ICLTensor *input, ICLTensor *topk_values, ICLTensor *topk_indices,
                               cl::Buffer *indices, cl::Buffer *temp_stack, int k, int n)
{
  ARM_COMPUTE_ERROR_ON(input == nullptr && indices == nullptr);
  ARM_COMPUTE_ERROR_ON(topk_values == nullptr && topk_indices == nullptr);
  ARM_COMPUTE_ERROR_ON(n == 0);

  _input = input;
  _topk_values = topk_values;
  _topk_indices = topk_indices;

  // Set kernel build options
  std::set<std::string> build_opts;

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("topkv2_quicksort", build_opts));

  unsigned int idx = 3 * num_arguments_per_1D_tensor();
  _kernel.setArg(idx++, *indices);
  _kernel.setArg(idx++, *temp_stack);
  _kernel.setArg<cl_int>(idx++, k);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, 1, 1));
  ICLKernel::configure_internal(win);
}

void CLTopKV2Single::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  unsigned int idx = 0;
  add_1D_tensor_argument(idx, _input, window);
  add_1D_tensor_argument(idx, _topk_values, window);
  add_1D_tensor_argument(idx, _topk_indices, window);

  enqueue(queue, *this, window);
}

////////////////////////////////////////////////////////////////////////////////
CLTopKV2Init::CLTopKV2Init() : _input(nullptr) {}

void CLTopKV2Init::configure(ICLTensor *input, cl::Buffer *in_key_buf, cl::Buffer *in_ind_buf,
                             int n)
{
  ARM_COMPUTE_ERROR_ON(input == nullptr && in_key_buf == nullptr);
  ARM_COMPUTE_ERROR_ON(in_ind_buf == nullptr);
  ARM_COMPUTE_ERROR_ON(n == 0);

  _input = input;

  // Set kernel build options
  std::set<std::string> build_opts;

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel("topkv2_init", build_opts));

  unsigned int idx = num_arguments_per_1D_tensor();
  _kernel.setArg(idx++, *in_key_buf);
  _kernel.setArg(idx++, *in_ind_buf);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, n, 1));
  ICLKernel::configure_internal(win);
}

void CLTopKV2Init::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  unsigned int idx = 0;
  add_1D_tensor_argument(idx, _input, window);

  enqueue(queue, *this, window);
}

////////////////////////////////////////////////////////////////////////////////
// This kernel makes a histogram of radix for each work item.
CLRadixSortHistogram::CLRadixSortHistogram() : _pass(0), _in_key_buf(nullptr) {}

void CLRadixSortHistogram::configure(cl::Buffer *hist_buf, int bits, int n)
{
  ARM_COMPUTE_ERROR_ON(hist_buf == nullptr);

  unsigned int radix = 1 << bits;
  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-D_BITS=" + support::cpp11::to_string(bits));
  build_opts.emplace("-D_RADIX=" + support::cpp11::to_string(radix));
  build_opts.emplace("-DPERMUT=1");

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("radixsort_histogram", build_opts));

  int loc_histo_size = radix * _ITEMS * sizeof(cl_int);

  unsigned int idx = 1;
  _kernel.setArg(idx++, *hist_buf);

  idx = 3;
  _kernel.setArg(idx++, loc_histo_size, nullptr);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, _GROUPS * _ITEMS, 1));
  ICLKernel::configure_internal(win);
}

void CLRadixSortHistogram::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  _kernel.setArg(0, *_in_key_buf);
  _kernel.setArg<cl_int>(2, _pass);

  cl::NDRange lws = cl::NDRange(_ITEMS, 1);

  enqueue(queue, *this, window, lws);
}

////////////////////////////////////////////////////////////////////////////////
CLRadixSortScanHistogram::CLRadixSortScanHistogram() {}

void CLRadixSortScanHistogram::configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits)
{
  ARM_COMPUTE_ERROR_ON(hist_buf == nullptr && glob_sum_buf == nullptr);

  unsigned int radix = 1 << bits;
  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-D_BITS=" + support::cpp11::to_string(bits));
  build_opts.emplace("-D_RADIX=" + support::cpp11::to_string(radix));
  build_opts.emplace("-DPERMUT=1");

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("radixsort_scanhistograms", build_opts));

  int temp_size =
      std::max<uint32_t>(_HISTOSPLIT, _ITEMS * _GROUPS * radix / _HISTOSPLIT) * sizeof(cl_uint);

  unsigned int idx = 0;
  _kernel.setArg(idx++, *hist_buf);
  _kernel.setArg(idx++, temp_size, nullptr);
  _kernel.setArg(idx++, *glob_sum_buf);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, radix * _GROUPS * _ITEMS / 2, 1));
  ICLKernel::configure_internal(win);
}

void CLRadixSortScanHistogram::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  const unsigned int gws_x = (window.x().end() - window.x().start()) / window.x().step();
  cl::NDRange lws = cl::NDRange(gws_x / _HISTOSPLIT, 1);

  enqueue(queue, *this, window, lws);
}

////////////////////////////////////////////////////////////////////////////////
CLRadixSortGlobalScanHistogram::CLRadixSortGlobalScanHistogram() {}

void CLRadixSortGlobalScanHistogram::configure(cl::Buffer *glob_sum_buf, cl::Buffer *temp_buf,
                                               int bits)
{
  ARM_COMPUTE_ERROR_ON(glob_sum_buf == nullptr && temp_buf == nullptr);

  unsigned int radix = 1 << bits;
  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-D_BITS=" + support::cpp11::to_string(bits));
  build_opts.emplace("-D_RADIX=" + support::cpp11::to_string(radix));
  build_opts.emplace("-DPERMUT=1");

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("radixsort_scanhistograms", build_opts));

  int temp_size =
      std::max<uint32_t>(_HISTOSPLIT, _ITEMS * _GROUPS * radix / _HISTOSPLIT) * sizeof(cl_uint);

  unsigned int idx = 0;
  _kernel.setArg(idx++, *glob_sum_buf);
  _kernel.setArg(idx++, temp_size, nullptr);
  _kernel.setArg(idx++, *temp_buf);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, _HISTOSPLIT / 2, 1));
  ICLKernel::configure_internal(win);
}

void CLRadixSortGlobalScanHistogram::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  const unsigned int gws_x = (window.x().end() - window.x().start()) / window.x().step();
  cl::NDRange lws = cl::NDRange(gws_x, 1);

  enqueue(queue, *this, window, lws);
}

////////////////////////////////////////////////////////////////////////////////
CLRadixSortPasteHistogram::CLRadixSortPasteHistogram() {}

void CLRadixSortPasteHistogram::configure(cl::Buffer *hist_buf, cl::Buffer *glob_sum_buf, int bits)
{
  ARM_COMPUTE_ERROR_ON(hist_buf == nullptr && glob_sum_buf == nullptr);

  unsigned int radix = 1 << bits;
  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-D_BITS=" + support::cpp11::to_string(bits));
  build_opts.emplace("-D_RADIX=" + support::cpp11::to_string(radix));
  build_opts.emplace("-DPERMUT=1");

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("radixsort_pastehistograms", build_opts));

  unsigned int idx = 0;
  _kernel.setArg(idx++, *hist_buf);
  _kernel.setArg(idx++, *glob_sum_buf);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, radix * _GROUPS * _ITEMS / 2, 1));
  ICLKernel::configure_internal(win);
}

void CLRadixSortPasteHistogram::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  const unsigned int gws_x = (window.x().end() - window.x().start()) / window.x().step();
  cl::NDRange lws = cl::NDRange(gws_x / _HISTOSPLIT, 1);

  enqueue(queue, *this, window, lws);
}

////////////////////////////////////////////////////////////////////////////////
CLRadixSortReorder::CLRadixSortReorder()
    : _pass(0), _in_key_buf(nullptr), _out_key_buf(nullptr), _in_ind_buf(nullptr),
      _out_ind_buf(nullptr)
{
}

void CLRadixSortReorder::configure(cl::Buffer *hist_buf, int bits, int n)
{
  ARM_COMPUTE_ERROR_ON(hist_buf == nullptr);
  ARM_COMPUTE_ERROR_ON(n == 0);

  unsigned int radix = 1 << bits;
  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-D_BITS=" + support::cpp11::to_string(bits));
  build_opts.emplace("-D_RADIX=" + support::cpp11::to_string(radix));
  build_opts.emplace("-DPERMUT=1");

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("radixsort_reorder", build_opts));

  unsigned int idx = 2;
  _kernel.setArg(idx++, *hist_buf);

  idx = 6;
  _kernel.setArg(idx++, sizeof(uint) * radix * _ITEMS, nullptr);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, _GROUPS * _ITEMS, 1));
  ICLKernel::configure_internal(win);
}

void CLRadixSortReorder::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  const unsigned int gws_x = (window.x().end() - window.x().start()) / window.x().step();
  unsigned int lx = std::max(1U, (gws_x / _HISTOSPLIT));
  cl::NDRange lws = (lx < gws_x) ? cl::NDRange(lx, 1) : cl::NDRange(1, 1);

  _kernel.setArg(0, *_in_key_buf);
  _kernel.setArg(1, *_out_key_buf);
  _kernel.setArg<cl_int>(3, _pass);
  _kernel.setArg(4, *_in_ind_buf);
  _kernel.setArg(5, *_out_ind_buf);

  enqueue(queue, *this, window, lws);
}

////////////////////////////////////////////////////////////////////////////////
CLTopKV2FindFirstNegative::CLTopKV2FindFirstNegative() : _out_key_buf(nullptr) {}

void CLTopKV2FindFirstNegative::configure(cl::Buffer *first_negative_idx_buf, int n)
{
  ARM_COMPUTE_ERROR_ON(first_negative_idx_buf == nullptr);
  ARM_COMPUTE_ERROR_ON(n == 0);

  // Set kernel build options
  std::set<std::string> build_opts;

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("topkv2_find_first_negative", build_opts));

  unsigned int idx = 1;
  _kernel.setArg(idx++, *first_negative_idx_buf);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, n, 1));
  ICLKernel::configure_internal(win);
}

void CLTopKV2FindFirstNegative::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  unsigned int idx = 0;
  _kernel.setArg(idx++, *_out_key_buf);

  enqueue(queue, *this, window);
}

////////////////////////////////////////////////////////////////////////////////
CLTopKV2ReorderNegatives::CLTopKV2ReorderNegatives()
    : _in_key_buf(nullptr), _out_key_buf(nullptr), _in_ind_buf(nullptr), _out_ind_buf(nullptr)
{
}

void CLTopKV2ReorderNegatives::configure(cl::Buffer *first_negative_idx_buf, int n)
{
  ARM_COMPUTE_ERROR_ON(first_negative_idx_buf == nullptr);
  ARM_COMPUTE_ERROR_ON(n == 0);

  // Set kernel build options
  std::set<std::string> build_opts;

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("topkv2_reorder_negatives", build_opts));

  unsigned int idx = 4;
  _kernel.setArg(idx++, *first_negative_idx_buf);
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, n, 1));
  ICLKernel::configure_internal(win);
}

void CLTopKV2ReorderNegatives::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  unsigned int idx = 0;
  _kernel.setArg(idx++, *_in_key_buf);
  _kernel.setArg(idx++, *_out_key_buf);
  _kernel.setArg(idx++, *_in_ind_buf);
  _kernel.setArg(idx++, *_out_ind_buf);

  enqueue(queue, *this, window);
}

////////////////////////////////////////////////////////////////////////////////
CLTopKV2Store::CLTopKV2Store()
    : _values(nullptr), _indices(nullptr), _out_key_buf(nullptr), _out_ind_buf(nullptr)
{
}

void CLTopKV2Store::configure(ICLTensor *values, ICLTensor *indices, int k, int n)
{
  ARM_COMPUTE_ERROR_ON(values == nullptr && indices == nullptr);
  ARM_COMPUTE_ERROR_ON(k == 0);
  ARM_COMPUTE_ERROR_ON(k > n);

  _values = values;
  _indices = indices;

  // Set kernel build options
  std::set<std::string> build_opts;

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel("topkv2_store", build_opts));

  unsigned int idx = 2 * num_arguments_per_1D_tensor() + 2;
  _kernel.setArg<cl_int>(idx++, n);

  // Configure kernel window
  Window win;
  win.set(0, Window::Dimension(0, k, 1));
  ICLKernel::configure_internal(win);
}

void CLTopKV2Store::setOutputBuffers(cl::Buffer *out_key_buf, cl::Buffer *out_ind_buf)
{
  _out_key_buf = out_key_buf;
  _out_ind_buf = out_ind_buf;
}

void CLTopKV2Store::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  unsigned int idx = 0;
  add_1D_tensor_argument(idx, _values, window);
  add_1D_tensor_argument(idx, _indices, window);
  _kernel.setArg(idx++, *_out_key_buf);
  _kernel.setArg(idx++, *_out_ind_buf);

  enqueue(queue, *this, window);
}

} // namespace arm_compute
#endif // Disable GPU implementation
