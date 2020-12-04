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

#include "arm_compute/runtime/CL/functions/CLTopKV2.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "arm_compute/core/CL/ICLTensor.h"

#include "../../topk_v2.h"

namespace arm_compute
{

CLTopKV2::CLTopKV2()
  : _k(0), _total_bits(0), _bits(0), _radix(0), _hist_buf_size(0), _glob_sum_buf_size(0), _n(0),
    _input(nullptr), _values(nullptr), _indices(nullptr), _qs_idx_buf(), _qs_temp_buf(),
    _hist_buf(), _glob_sum_buf(), _temp_buf(), _first_negative_idx_buf(), _in_key_buf(),
    _out_key_buf(), _in_ind_buf(), _out_ind_buf(), _p_in_key_buf(nullptr), _p_out_key_buf(nullptr),
    _p_in_ind_buf(nullptr), _p_out_ind_buf(nullptr) /*, _qs_kernel(),
    _init_kernel(), _hist_kernel(), _scan_hist_kernel(), _glob_scan_hist_kernel(),
    _paste_hist_kernel(), _reorder_kernel(), _find_first_negative_kernel(),
    _reorder_negatives_kernel(), _store_kernel()*/
{
}

void CLTopKV2::configure(ICLTensor *input, int k, ICLTensor *values, ICLTensor *indices,
                         int total_bits, int bits)
{
  _total_bits = total_bits;
  _bits = bits;
  _n = input->info()->tensor_shape()[0];

  // _total_bits should be divided by _bits.
  ARM_COMPUTE_ERROR_ON((_total_bits % _bits) != 0);

  _k = k;
  _radix = 1 << bits;

  _input = input;
  _values = values;
  _indices = indices;

  std::string topk_env;

// Disable GPU implementation
// TODO Enable GPU implementation with verification, or remove code
//      Invalid result on GPU
#if 0
  char *env = getenv("ACL_TOPKV2");
  if (env)
    topk_env = env;

  if (topk_env == "GPU_SINGLE")
  {
    _qs_idx_buf = cl::Buffer(CLScheduler::get().context(),
                             CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int) * _n);
    _qs_temp_buf = cl::Buffer(CLScheduler::get().context(),
                              CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int) * _n);

    _qs_kernel.configure(input, values, indices, &_qs_idx_buf, &_qs_temp_buf, k, _n);
  }
  else if (topk_env == "GPU")
  {
    // n should be divided by (_GROUPS * _ITEMS)
    ARM_COMPUTE_ERROR_ON((_n % (_GROUPS * _ITEMS)) != 0);

    _hist_buf_size = _radix * _GROUPS * _ITEMS;
    _glob_sum_buf_size = _HISTOSPLIT;

    _hist_buf = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(cl_int) * _hist_buf_size);
    _glob_sum_buf =
        cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                   sizeof(cl_int) * _glob_sum_buf_size);
    _temp_buf = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(cl_int) * _glob_sum_buf_size);
    _first_negative_idx_buf = cl::Buffer(CLScheduler::get().context(),
                                         CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int));
    _in_key_buf = cl::Buffer(CLScheduler::get().context(),
                             CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_float) * _n);
    _out_key_buf = cl::Buffer(CLScheduler::get().context(),
                              CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_float) * _n);
    _in_ind_buf = cl::Buffer(CLScheduler::get().context(),
                             CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int) * _n);
    _out_ind_buf = cl::Buffer(CLScheduler::get().context(),
                              CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int) * _n);

    _p_in_key_buf = &_in_key_buf;
    _p_out_key_buf = &_out_key_buf;
    _p_in_ind_buf = &_in_ind_buf;
    _p_out_ind_buf = &_out_ind_buf;

    _init_kernel.configure(input, _p_in_key_buf, _p_in_ind_buf, _n);
    _hist_kernel.configure(&_hist_buf, bits, _n);
    _scan_hist_kernel.configure(&_hist_buf, &_glob_sum_buf, bits);
    _glob_scan_hist_kernel.configure(&_glob_sum_buf, &_temp_buf, bits);
    _paste_hist_kernel.configure(&_hist_buf, &_glob_sum_buf, bits);
    _reorder_kernel.configure(&_hist_buf, bits, _n);
    _find_first_negative_kernel.configure(&_first_negative_idx_buf, _n);
    _reorder_negatives_kernel.configure(&_first_negative_idx_buf, _n);
    _store_kernel.configure(values, indices, k, _n);
  }
  else
#endif // Disable GPU implementation
  {
    // DO NOTHING for CPU.
  }
}

void CLTopKV2::run()
{
  std::string topk_env;
#if 0
  char *env = getenv("ACL_TOPKV2");
  if (env)
    topk_env = env;

  if (topk_env == "GPU_SINGLE")
  {
    run_on_gpu_single_quicksort();
  }
  else if (topk_env == "GPU")
  {
    run_on_gpu();
  }
  else
#endif
  {
    run_on_cpu();
  }
}

#if 0
void CLTopKV2::run_on_gpu_single_quicksort()
{
  // This is a single threaded quick sort implementation.
  CLScheduler::get().enqueue(_qs_kernel, false);

  arm_compute::CLScheduler::get().sync();
}

void CLTopKV2::run_on_gpu()
{
  cl::CommandQueue q = CLScheduler::get().queue();

  // 1. CLTopKV2Init set key buffer and index buffer.
  //  - Key buffer is set as the same value of the layer's input
  //  - Values in the index buffer are set as their indices.
  CLScheduler::get().enqueue(_init_kernel, false);

  int n_passes = _total_bits / _bits;

  // 2. Repeat (total_bits/bits) times.
  //   - total_bits is the number of bits of the data type (e.g., 32 for float)
  //   - bits defines number of buckets (e.g. 16 buckets where bit is 4)
  for (int pass = 0; pass < n_passes; ++pass)
  {
    arm_compute::CLScheduler::get().sync();

    // 2.1. Calculate histogram with _GROUPS * _ITEMS threads
    _hist_kernel.setPass(pass, _p_in_key_buf);
    CLScheduler::get().enqueue(_hist_kernel, false);

    // 2.2. Calculate prefix sum locally with multiple threads
    CLScheduler::get().enqueue(_scan_hist_kernel, false);
    // 2.3. Calculate prefix sum within a work group
    CLScheduler::get().enqueue(_glob_scan_hist_kernel, false);
    // 2.4. Calculate global prefix sum
    CLScheduler::get().enqueue(_paste_hist_kernel, false);

    // 2.5. Reorder keys and indices based on the global prefix sum
    _reorder_kernel.setPass(pass, _p_in_key_buf, _p_out_key_buf, _p_in_ind_buf, _p_out_ind_buf);
    CLScheduler::get().enqueue(_reorder_kernel, false);

    cl::Buffer *tmp;
    // swap key buffers
    tmp = _p_in_key_buf;
    _p_in_key_buf = _p_out_key_buf;
    _p_out_key_buf = tmp;

    // swap index buffers
    tmp = _p_in_ind_buf;
    _p_in_ind_buf = _p_out_ind_buf;
    _p_out_ind_buf = tmp;
  }

  // 3. Get the first negative index
  // Because we swap in_buf and out_buf at the end of the above for loop,
  // the output buffers are in bufs.
  _find_first_negative_kernel.setOutputBuffer(_p_in_key_buf);
  CLScheduler::get().enqueue(_find_first_negative_kernel, false);

  // 4. Correct odering of negatives
  //   - Since radix sort does not consider negatives, negatives are considered as bigger values
  //   than positives.
  // reordered data will be stored in _p_out_key_buf and _p_out_ind_buf
  _reorder_negatives_kernel.setBuffers(_p_in_key_buf, _p_out_key_buf, _p_in_ind_buf,
                                       _p_out_ind_buf);
  CLScheduler::get().enqueue(_reorder_negatives_kernel, false);

  // 5. Extract top k values from sorted keys and indices.
  _store_kernel.setOutputBuffers(_p_out_key_buf, _p_out_ind_buf);
  CLScheduler::get().enqueue(_store_kernel, false);

  arm_compute::CLScheduler::get().sync();

#if 0
  // below code is left for debugging.
  int first_neg;
  q.enqueueReadBuffer(_first_negative_idx_buf, CL_TRUE, 0, sizeof(cl_int), &first_neg);
  std::cout << "first neg = " << first_neg << std::endl;

  float in_key[_n];
  q.enqueueReadBuffer(*_p_in_key_buf, CL_TRUE, 0, sizeof(cl_float)*_n, in_key);
  for(uint32_t i = 0 ; i < _n; ++i) {
    std::cout << "in_key[" << i << "] = " << in_key[i] << std::endl;
  }

  float out_key[_n];
  q.enqueueReadBuffer(*_p_out_key_buf, CL_TRUE, 0, sizeof(cl_float)*_n, out_key);
  for(uint32_t i = 0 ; i < _n; ++i) {
    std::cout << "out_key[" << i << "] = " << out_key[i] << std::endl;
  }

  int in_ind[_n];
  q.enqueueReadBuffer(*_p_in_ind_buf, CL_TRUE, 0, sizeof(cl_int)*_n, in_ind);
  for(uint32_t i = 0 ; i < _n; ++i) {
    std::cout << "in_ind[" << i << "] = " << in_ind[i] << std::endl;
  }

  int out_ind[_n];
  q.enqueueReadBuffer(*_p_out_ind_buf, CL_TRUE, 0, sizeof(cl_int)*_n, out_ind);
  for(uint32_t i = 0 ; i < _n; ++i) {
    std::cout << "out_ind[" << i << "] = " << out_ind[i] << std::endl;
  }

  int hist_buf[_hist_buf_size];
  q.enqueueReadBuffer(_hist_buf, CL_TRUE, 0, sizeof(cl_int)*_hist_buf_size, hist_buf);
  for(uint32_t i = 0 ; i < _hist_buf_size; ++i) {
    std::cout << "hist_buf[" << i << "] = " << hist_buf[i] << std::endl;
  }

  int glob_sum_buf[_glob_sum_buf_size];
  q.enqueueReadBuffer(_glob_sum_buf, CL_TRUE, 0, sizeof(cl_int)*_glob_sum_buf_size, glob_sum_buf);
  for(uint32_t i = 0 ; i < _glob_sum_buf_size; ++i) {
    std::cout << "glob_sum_buf[" << i << "] = " << glob_sum_buf[i] << std::endl;
  }

#endif
}
#endif // Disable GPU implementation

void CLTopKV2::run_on_cpu()
{
  cl::CommandQueue q = CLScheduler::get().queue();
  // const Window& w = _topkv2_kernel.window();

  _input->map(q);
  _values->map(q);
  _indices->map(q);

  // int row_size = (w[0].end() - w[0].start()) / w[0].step();
  int row_size = _input->info()->tensor_shape()[0];
  int rank = _input->info()->num_dimensions();

  if (rank > 2)
    throw std::runtime_error("Not supported type.");

  int row_num = (rank == 2 ? _input->info()->tensor_shape()[1] : 1);

  if (_input->info()->data_type() == DataType::F32)
  {
    nnfw::rt::optimized_ops::TopK<float>(row_size, row_num, (float *)_input->buffer(), _k,
                                         (int32 *)_indices->buffer(), (float *)_values->buffer());
  }
  else if (_input->info()->data_type() == DataType::S32)
  {
    nnfw::rt::optimized_ops::TopK<int32_t>(row_size, row_num, (int32_t *)_input->buffer(), _k,
                                           (int32 *)_indices->buffer(),
                                           (int32_t *)_values->buffer());
  }
  else if (_input->info()->data_type() == DataType::QASYMM8)
  {
    nnfw::rt::optimized_ops::TopK<uint8_t>(row_size, row_num, (uint8_t *)_input->buffer(), _k,
                                           (int32 *)_indices->buffer(),
                                           (uint8_t *)_values->buffer());
  }
  else
  {
    throw std::runtime_error("Not supported type.");
  }

  _input->unmap(q);
  _values->unmap(q);
  _indices->unmap(q);
}

} // namespace arm_compute
