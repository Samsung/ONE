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

__global inline float *get_vec_elem(Vector *vec, int idx)
{
  return (__global float *)(vec->ptr + idx * vec->stride_x);
}

__global inline int *get_vec_elem_int(Vector *vec, int idx)
{
  return (__global int *)(vec->ptr + idx * vec->stride_x);
}

// A utility function to swap two elements
void swap(__global float *a, __global float *b)
{
  float t = *a;
  *a = *b;
  *b = t;
}

void swap_idx(__global int *a, __global int *b)
{
  int t = *a;
  *a = *b;
  *b = t;
}

/* This function is same in both iterative and recursive*/
int partition(Vector *arr, __global int *indices, int l, int h)
{
  float x = *get_vec_elem(arr, h);
  int i = (l - 1);

  for (int j = l; j <= h - 1; j++)
  {
    if (*get_vec_elem(arr, j) >= x)
    {
      i++;
      swap(get_vec_elem(arr, i), get_vec_elem(arr, j));
      swap_idx(&indices[i], &indices[j]);
    }
  }
  swap(get_vec_elem(arr, i + 1), get_vec_elem(arr, h));
  swap_idx(&indices[i + 1], &indices[h]);
  return (i + 1);
}

/* A[] --> Array to be sorted,
   l  --> Starting index,
   h  --> Ending index */
void quickSortIterative(Vector *arr, __global int *indices, __global int *stack, int l, int h)
{
  // Create an auxiliary stack

  // initialize top of stack
  int top = -1;

  // push initial values of l and h to stack
  stack[++top] = l;
  stack[++top] = h;

  // Keep popping from stack while is not empty
  while (top >= 0)
  {
    // Pop h and l
    h = stack[top--];
    l = stack[top--];

    // Set pivot element at its correct position
    // in sorted array
    int p = partition(arr, indices, l, h);

    // If there are elements on left side of pivot,
    // then push left side to stack
    if (p - 1 > l)
    {
      stack[++top] = l;
      stack[++top] = p - 1;
    }

    // If there are elements on right side of pivot,
    // then push right side to stack
    if (p + 1 < h)
    {
      stack[++top] = p + 1;
      stack[++top] = h;
    }
  }
}

__kernel void topkv2_quicksort(VECTOR_DECLARATION(input), VECTOR_DECLARATION(topk_values),
                               VECTOR_DECLARATION(topk_indices), __global int *indices,
                               __global int *temp_stack, int k, int n)
{
  Vector input = CONVERT_TO_VECTOR_STRUCT_NO_STEP(input);
  Vector topk_values = CONVERT_TO_VECTOR_STRUCT_NO_STEP(topk_values);
  Vector topk_indices = CONVERT_TO_VECTOR_STRUCT_NO_STEP(topk_indices);

  for (int i = 0; i < n; ++i)
  {
    indices[i] = i;
  }

  quickSortIterative(&input, indices, temp_stack, 0, n - 1);

  // extract k items.
  for (int i = 0; i < k; ++i)
  {
    *get_vec_elem(&topk_values, i) = *get_vec_elem(&input, i);
    *get_vec_elem_int(&topk_indices, i) = indices[i];
  }
}
