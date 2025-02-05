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

#include "helpers.h"

__kernel void topkv2_init(VECTOR_DECLARATION(input), __global float *in_key_buf,
                          __global int *in_ind_buf, const int n)
{
  int gid = get_global_id(0);
  int lws = get_local_size(0);
  int groups = get_num_groups(0);
  int gws = lws * groups;
  int iter = n / gws;

  Vector input = CONVERT_TO_VECTOR_STRUCT_NO_STEP(input);

  for (int i = 0; i < iter; ++i)
  {
    int idx = i * gws + gid;
    in_key_buf[idx] = *(__global float *)(input.ptr + idx * input.stride_x);
    in_ind_buf[idx] = idx;
  }
}

__kernel void topkv2_find_first_negative(__global float *out_key_buf,
                                         __global int *first_negative_idx, int n)
{
  int gid = get_global_id(0);

  if (gid == n - 1)
  {
    // if the last item is positive, the first negative index is n.
    if (out_key_buf[gid] > 0.f)
      *first_negative_idx = n;
  }
  else if (gid == 0)
  {
    // if the first item is negative, set it 0.
    if (out_key_buf[gid] < 0.f)
      *first_negative_idx = 0;
  }
  else
  {
    // if its left is positive and it is negative, then it is the first negative item.
    if (out_key_buf[gid - 1] > 0.f && out_key_buf[gid] < 0.f)
      *first_negative_idx = gid;
  }
}

__kernel void topkv2_reorder_negatives(__global float *in_key_buf, __global float *out_key_buf,
                                       __global float *in_ind_buf, __global float *out_ind_buf,
                                       __global int *first_negative_idx, int n)
{
  int gid = get_global_id(0);

  int num_negs = n - *first_negative_idx;
  int in_idx;

  if (gid < num_negs)
  {
    in_idx = n - 1 - gid;
  }
  else
  {
    in_idx = gid - num_negs;
  }

  out_key_buf[gid] = in_key_buf[in_idx];
  out_ind_buf[gid] = in_ind_buf[in_idx];
}

__kernel void topkv2_store(VECTOR_DECLARATION(values), VECTOR_DECLARATION(indices),
                           __global float *out_key_buf, __global int *out_ind_buf, int n)
{
  int gid = get_global_id(0);

  Vector values = CONVERT_TO_VECTOR_STRUCT_NO_STEP(values);
  Vector indices = CONVERT_TO_VECTOR_STRUCT_NO_STEP(indices);

  int idx = n - 1 - gid;

  *(__global float *)(values.ptr + gid * values.stride_x) = out_key_buf[idx];
  *(__global int *)(indices.ptr + gid * indices.stride_x) = out_ind_buf[idx];
}
