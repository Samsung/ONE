/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_TRANSPOSE_H__
#define __NNFW_CKER_TRANSPOSE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{
namespace reference
{

template <typename T>
void TransposeImpl(const TransposeParams &params, const Shape &unextended_input_shape,
                   const T *input_data, const Shape &unextended_output_shape, T *output_data)
{
  const int unextended_output_size = unextended_output_shape.DimensionsCount();
  assert(unextended_input_shape.DimensionsCount() <= 4);
  assert(unextended_output_size <= 4);
  assert(unextended_output_size == params.perm_count);
  const Shape input_shape = Shape::ExtendedShape(4, unextended_input_shape);
  const Shape output_shape = Shape::ExtendedShape(4, unextended_output_shape);
  const int input_ext_size = 4 - unextended_input_shape.DimensionsCount();
  const int output_ext_size = 4 - unextended_output_size;

  // The perm data is extended to match the output, each index incremented by
  // the amount of front padding of the input shape.
  int extended_perm[4];
  for (int i = 0; i < output_ext_size; ++i)
  {
    extended_perm[i] = i;
  }
  for (int i = 0; i < unextended_output_size; ++i)
  {
    extended_perm[i + output_ext_size] = params.perm[i] + input_ext_size;
  }

  int out_sizes[4];
  // Compute the inverse permutation array so we can do an output centered
  // transpose. Also, check to make sure output_dims is matching input_dims.
  for (int k = 0; k < 4; k++)
  {
    out_sizes[k] = MatchingDim(input_shape, extended_perm[k], output_shape, k);
  }

  // Naive transpose loop (iterate on output index and compute input index).
  int o[4]; // loop index (on output).
  int i[4];
  for (o[3] = 0; o[3] < out_sizes[3]; o[3]++)
  {
    i[extended_perm[3]] = o[3];
    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++)
    {
      i[extended_perm[2]] = o[2];
      for (o[1] = 0; o[1] < out_sizes[1]; o[1]++)
      {
        i[extended_perm[1]] = o[1];
        for (o[0] = 0; o[0] < out_sizes[0]; o[0]++)
        {
          i[extended_perm[0]] = o[0];
          output_data[Offset(output_shape, o)] = input_data[Offset(input_shape, i)];
        }
      }
    }
  }
}

template <typename T>
void Transpose(const TransposeParams &params, const Shape &unextended_input_shape,
               const T *input_data, const Shape &unextended_output_shape, T *output_data)
{
  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (sizeof(T))
  {
    case 1:
      TransposeImpl<int8_t>(params, unextended_input_shape,
                            reinterpret_cast<const int8_t *>(input_data), unextended_output_shape,
                            reinterpret_cast<int8_t *>(output_data));
      break;
    case 2:
      TransposeImpl<int16_t>(params, unextended_input_shape,
                             reinterpret_cast<const int16_t *>(input_data), unextended_output_shape,
                             reinterpret_cast<int16_t *>(output_data));
      break;

    case 4:
      TransposeImpl<int32_t>(params, unextended_input_shape,
                             reinterpret_cast<const int32_t *>(input_data), unextended_output_shape,
                             reinterpret_cast<int32_t *>(output_data));
      break;
    case 8:
      TransposeImpl<int64_t>(params, unextended_input_shape,
                             reinterpret_cast<const int64_t *>(input_data), unextended_output_shape,
                             reinterpret_cast<int64_t *>(output_data));
      break;
  }
}
} // namespace reference

namespace
{

bool IsTranspose2DApplicable(const TransposeParams &params, const Shape &input_shape, int *dim0,
                             int *dim1)
{
  const int dims_cnt = input_shape.DimensionsCount();

  if (dims_cnt == 2)
  {
    *dim0 = input_shape.Dims(0);
    *dim1 = input_shape.Dims(1);
    return true;
  }

  const int first_perm = params.perm[0];
  for (int i = 1; i < dims_cnt; ++i)
  {
    int rebased = params.perm[i] - first_perm;
    if (rebased < 0)
    {
      rebased += dims_cnt;
    }
    if (rebased != i)
    {
      return false;
    }
  }
  *dim0 = 1;
  *dim1 = 1;
  for (int i = 0; i < dims_cnt; ++i)
  {
    if (i < first_perm)
    {
      *dim0 *= input_shape.Dims(i);
    }
    else
    {
      *dim1 *= input_shape.Dims(i);
    }
  }
  return true;
}

void RemoveOneSizeDimensions(Shape *input_shape, Shape *output_shape, TransposeParams *params)
{
  const int dims_cnt = input_shape->DimensionsCount();
  assert(params->perm_count == dims_cnt);

  bool foundOneSizeDim = false;
  for (int i = 0; i < dims_cnt; ++i)
  {
    if (input_shape->Dims(i) == 1)
    {
      foundOneSizeDim = true;
      break;
    }
  }

  // Return here if there is no one size dimension.
  if (!foundOneSizeDim)
    return;

  // Handle the case where all the dimension size is one.
  if (input_shape->FlatSize() == 1)
  {
    input_shape->Resize(1);
    input_shape->SetDim(0, 1);
    output_shape->Resize(1);
    output_shape->SetDim(0, 1);
    params->perm_count = 1;
    params->perm[0] = 0;
    return;
  }

  // Resize input shape.
  int new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i)
  {
    if (input_shape->Dims(i) == 1)
    {
      continue;
    }
    input_shape->SetDim(new_dims_cnt, input_shape->Dims(i));
    ++new_dims_cnt;
  }
  input_shape->Resize(new_dims_cnt);

  // Resize output shape and re-calculate the perm parameter.
  TransposeParams new_params;
  new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i)
  {
    if (output_shape->Dims(i) == 1)
    {
      continue;
    }
    new_params.perm[new_dims_cnt] = params->perm[i];
    output_shape->SetDim(new_dims_cnt, output_shape->Dims(i));
    ++new_dims_cnt;
  }
  output_shape->Resize(new_dims_cnt);
  new_params.perm_count = new_dims_cnt;

  for (int i = 0; i < new_dims_cnt; ++i)
  {
    int min_val_idx = -1;
    for (int j = 0; j < new_dims_cnt; ++j)
    {
      if (new_params.perm[j] >= i &&
          (min_val_idx == -1 || new_params.perm[min_val_idx] > new_params.perm[j]))
      {
        min_val_idx = j;
      }
    }
    new_params.perm[min_val_idx] = i;
  }
  *params = new_params;
}

size_t Flatten(const Shape &input_shape, const Shape &output_shape, const TransposeParams &params,
               Shape *non_flatten_input_shape, Shape *non_flatten_output_shape,
               TransposeParams *non_flatten_params)
{
  // Calculate the total size of non-flatten dimensions.
  int skip_dims_cnt = 0;
  size_t flat_size = input_shape.FlatSize();
  for (int i = 0; i < params.perm_count; ++i)
  {
    if (params.perm[i] == i)
    {
      flat_size /= input_shape.Dims(i);
      ++skip_dims_cnt;
    }
    else
    {
      break;
    }
  }

  // Shrink the shapes and re-calculate the perm parameter.
  const int new_dims_cnt = params.perm_count - skip_dims_cnt;
  non_flatten_input_shape->Resize(new_dims_cnt);
  non_flatten_output_shape->Resize(new_dims_cnt);
  non_flatten_params->perm_count = new_dims_cnt;

  for (int i = skip_dims_cnt; i < params.perm_count; ++i)
  {
    non_flatten_input_shape->SetDim(i - skip_dims_cnt, input_shape.Dims(i));
    non_flatten_output_shape->SetDim(i - skip_dims_cnt, output_shape.Dims(i));
    non_flatten_params->perm[i - skip_dims_cnt] = params.perm[i];
  }
  for (int i = 0; i < new_dims_cnt; ++i)
  {
    int min_val_idx = -1;
    for (int j = 0; j < new_dims_cnt; ++j)
    {
      if (non_flatten_params->perm[j] >= i &&
          (min_val_idx == -1 ||
           non_flatten_params->perm[min_val_idx] > non_flatten_params->perm[j]))
      {
        min_val_idx = j;
      }
    }
    non_flatten_params->perm[min_val_idx] = i;
  }

  return flat_size;
}

} // namespace

// Transpose2D only deals with typical 2D matrix transpose ops.
// Perform transpose by transposing 4x4 blocks of the input, proceeding from
// left to right (down the rows) of the input, and then from top to bottom.
template <typename T>
inline void Transpose2D(const Shape &input_shape, const T *input_data, const Shape &output_shape,
                        T *output_data)
{
  assert(input_shape.DimensionsCount() == 2);
  assert(output_shape.DimensionsCount() == 2);
  UNUSED_RELEASE(output_shape);

  const int d0 = input_shape.DimsData()[0];
  const int d1 = input_shape.DimsData()[1];
  const int kLines = 4;
  const int kSkipSize = (kLines - 1) * d1;

  const T *input = input_data;

  int i = 0;
  for (; i <= d0 - kLines; i += kLines)
  {
    T *output = output_data + i;

    const T *input_ptr = input;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);

    int j = 0;
    for (; j <= d1 - kLines; j += kLines)
    {
      input_ptr = input;
      const T a00 = input_ptr[0];
      const T a01 = input_ptr[1];
      const T a02 = input_ptr[2];
      const T a03 = input_ptr[3];
      input_ptr += d1;
      const T a10 = input_ptr[0];
      const T a11 = input_ptr[1];
      const T a12 = input_ptr[2];
      const T a13 = input_ptr[3];
      input_ptr += d1;
      const T a20 = input_ptr[0];
      const T a21 = input_ptr[1];
      const T a22 = input_ptr[2];
      const T a23 = input_ptr[3];
      input_ptr += d1;
      const T a30 = input_ptr[0];
      const T a31 = input_ptr[1];
      const T a32 = input_ptr[2];
      const T a33 = input_ptr[3];

      output[0] = a00;
      output[1] = a10;
      output[2] = a20;
      output[3] = a30;
      output += d0;

      output[0] = a01;
      output[1] = a11;
      output[2] = a21;
      output[3] = a31;
      output += d0;

      output[0] = a02;
      output[1] = a12;
      output[2] = a22;
      output[3] = a32;
      output += d0;

      output[0] = a03;
      output[1] = a13;
      output[2] = a23;
      output[3] = a33;
      output += d0;

      input += kLines;
    }
    if (j == d1)
    {
      input += kSkipSize;
    }
    else
    {
      for (int p = 0; p < kLines; ++p)
      {
        for (int q = 0; q < d1 - j; ++q)
        {
          *(output + q * d0 + p) = *(input + p * d1 + q);
        }
      }
      input += (d1 - j) + kSkipSize;
    }
  }
  for (; i < d0; ++i)
  {
    T *output = output_data + i;
    for (int j = 0; j < d1; ++j)
    {
      *output = *input;
      output += d0;
      ++input;
    }
  }
}

// TODO(alanchiao): see if we can reduce the number
// of lines of code in branching without affecting latency.
template <typename T>
inline void Transpose3D(const TransposeParams &params, const Shape &input_shape,
                        const T *input_data, const Shape &, T *output_data)
{
  int s2, s3;
  s2 = input_shape.Dims(1);
  s3 = input_shape.Dims(2);

  int p1 = 0;
  int p2 = 0;
  int p3 = 0;

  if (params.perm[0] == 2)
  {
    p1 = 1;
  }
  else if (params.perm[1] == 2)
  {
    p2 = 1;
  }
  else
  {
    p3 = 1;
  }

  if (params.perm[0] == 1)
  {
    p1 = s3;
  }
  else if (params.perm[1] == 1)
  {
    p2 = s3;
  }
  else
  {
    p3 = s3;
  }

  if (params.perm[0] == 0)
  {
    p1 = s2 * s3;
  }
  else if (params.perm[1] == 0)
  {
    p2 = s2 * s3;
  }
  else
  {
    p3 = s2 * s3;
  }

  int o_s[3];
  o_s[0] = input_shape.Dims(params.perm[0]);
  o_s[1] = input_shape.Dims(params.perm[1]);
  o_s[2] = input_shape.Dims(params.perm[2]);

  for (int i1 = 0; i1 < o_s[0]; ++i1)
  {
    for (int i2 = 0; i2 < o_s[1]; ++i2)
    {
      for (int i3 = 0; i3 < o_s[2]; ++i3)
      {
        const int i = i1 * p1 + i2 * p2 + i3 * p3;
        const int o = i1 * o_s[1] * o_s[2] + i2 * o_s[2] + i3;
        output_data[o] = input_data[i];
      }
    }
  }
}

template <typename T>
void TransposeImpl(const TransposeParams &params, const Shape &input_shape, const T *input_data,
                   const Shape &output_shape, T *output_data)
{
  const int dims_cnt = input_shape.DimensionsCount();

  int dim0, dim1;
  if (IsTranspose2DApplicable(params, input_shape, &dim0, &dim1))
  {
    Transpose2D(Shape({dim0, dim1}), input_data, Shape({dim1, dim0}), output_data);
    return;
  }

  // TODO(b/141217325): notably Eigen is better suited for
  // larger inputs whereas Transpose3D is generally
  // better for smaller ones.
  //
  // E.g. on Nexus 5, Eigen is better for size 96^3 and up
  // and Transpose3D is better for 72^3 and down.
  //
  // 96^3 is not mobile-friendly for certain usecases
  // (e.g. model used in beam search for seq2seq) but is in others.
  // Consider tradeoffs.
  if (dims_cnt == 3)
  {
    Transpose3D(params, input_shape, input_data, output_shape, output_data);
    return;
  }

  // Reroute to the reference version if an optimized method for the given data
  // is not available.
  reference::Transpose(params, input_shape, input_data, output_shape, output_data);
}

template <typename T>
void Transpose(const TransposeParams &unshrunk_params, const Shape &unshrunk_input_shape,
               const T *input_data, const Shape &unshrunk_output_shape, T *output_data)
{
  const int output_size = unshrunk_output_shape.DimensionsCount();
  assert(unshrunk_input_shape.DimensionsCount() <= 4);
  assert(output_size <= 4);
  assert(output_size == unshrunk_params.perm_count);

  Shape shrunk_input_shape = Shape(unshrunk_input_shape);

  Shape shrunk_output_shape = Shape(unshrunk_output_shape);

  TransposeParams shrunk_params = unshrunk_params;

  // Reduce any dimensions that have one size. Lower transpose op usually
  // performs better since memory access patterns will be improved.
  RemoveOneSizeDimensions(&shrunk_input_shape, &shrunk_output_shape, &shrunk_params);

  // Handle identity cases.
  // TODO(b/140779653): Add an optimization pass in the conversion process to
  // remove transpose op nodes where they do nothing like the below one.
  bool identical = true;
  for (int i = 0; i < shrunk_params.perm_count; ++i)

  {
    if (shrunk_params.perm[i] != i)

    {
      identical = false;
      break;
    }
  }
  if (identical)
  {
    memcpy(output_data, input_data, unshrunk_input_shape.FlatSize() * sizeof(T));
    return;
  }

  // Reduce dimensions by flattening.
  if (shrunk_params.perm[0] == 0 && output_size >= 3)

  {
    Shape non_flatten_input_shape;
    Shape non_flatten_output_shape;
    TransposeParams non_flatten_params;
    const int total_size = shrunk_input_shape.FlatSize();

    const int non_flatten_size =
        Flatten(shrunk_input_shape, shrunk_output_shape, shrunk_params,

                &non_flatten_input_shape, &non_flatten_output_shape, &non_flatten_params);
    assert(non_flatten_params.perm[0] != 0);

    for (int i = 0; i < total_size; i += non_flatten_size)
    {
      TransposeImpl(non_flatten_params, non_flatten_input_shape, input_data + i,
                    non_flatten_output_shape, output_data + i);
    }
    return;
  }

  // Call non-flattened case.
  TransposeImpl(shrunk_params, shrunk_input_shape, input_data, shrunk_output_shape,

                output_data);
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRANSPOSE_H__
