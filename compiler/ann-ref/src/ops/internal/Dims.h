/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __DIMS_H__
#define __DIMS_H__

#include "Shape.h"
#include "Macro.h"

template <int N> struct Dims
{
  int sizes[N];
  int strides[N];
};

inline Dims<4> convertShapeToDims(const Shape &shape)
{
  Dims<4> dims;
  for (int i = 0; i < 4; i++)
  {
    dims.sizes[i] = 1;
  }

  if (shape.dimensions.size() == 1)
  {
    dims.sizes[0] = (int)getSizeOfDimension(shape, 0);
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      int src = (int)shape.dimensions.size() - i - 1;
      if (src >= 0)
      {
        dims.sizes[i] = (int)getSizeOfDimension(shape, src);
      }
    }
  }

  dims.strides[0] = 1;
  for (int i = 1; i < 4; i++)
  {
    dims.strides[i] = dims.strides[i - 1] * dims.sizes[i - 1];
  }
  return dims;
}

// From types.h in TensorFlow Lite
inline int Offset(const Dims<4> &dims, int i0, int i1, int i2, int i3)
{
  DCHECK(i0 >= 0 && i0 < dims.sizes[0]);
  DCHECK(i1 >= 0 && i1 < dims.sizes[1]);
  DCHECK(i2 >= 0 && i2 < dims.sizes[2]);
  DCHECK(i3 >= 0 && i3 < dims.sizes[3]);
  return i0 * dims.strides[0] + i1 * dims.strides[1] + i2 * dims.strides[2] + i3 * dims.strides[3];
}

// From types.h in TensorFlow Lite
//
// Get array size, DCHECKing that the dim index is in range.
template <int N> int ArraySize(const Dims<N> &array, int index)
{
  DCHECK(index >= 0 && index < N);
  return array.sizes[index];
}

// From types.h in TensorFlow Lite
template <int N> inline int FlatSize(const Dims<N> &dims)
{
  int flat_size = 1;
  for (int i = 0; i < N; ++i)
  {
    flat_size *= dims.sizes[i];
  }
  return flat_size;
}

// From types.h in TensorFlow Lite
inline int RequiredBufferSizeForDims(const Dims<4> &dims)
{
  int max_offset = 0;
  for (int i = 0; i < 4; i++)
  {
    max_offset += (dims.sizes[i] - 1) * dims.strides[i];
  }
  return max_offset + 1;
}

// From types.h in TensorFlow Lite
//
// Flat size calculation, checking that dimensions match with one or more other
// arrays.
template <int N> inline int MatchingFlatSize(const Dims<N> &dims, const Dims<N> &check_dims_0)
{
  for (int i = 0; i < N; ++i)
  {
    DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return FlatSize(dims);
}

// From types.h in TensorFlow Lite
template <int N>
inline int MatchingFlatSize(const Dims<N> &dims, const Dims<N> &check_dims_0,
                            const Dims<N> &check_dims_1)
{
  for (int i = 0; i < N; ++i)
  {
    DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return MatchingFlatSize(dims, check_dims_1);
}

// From types.h in TensorFlow Lite
template <int N>
inline int MatchingFlatSize(const Dims<N> &dims, const Dims<N> &check_dims_0,
                            const Dims<N> &check_dims_1, const Dims<N> &check_dims_2)
{
  for (int i = 0; i < N; ++i)
  {
    DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return FlatSize(dims, check_dims_1, check_dims_2);
}

// From types.h in TensorFlow Lite
template <int N>
inline int MatchingFlatSize(const Dims<N> &dims, const Dims<N> &check_dims_0,
                            const Dims<N> &check_dims_1, const Dims<N> &check_dims_2,
                            const Dims<N> &check_dims_3)
{
  for (int i = 0; i < N; ++i)
  {
    DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return FlatSize(dims, check_dims_1, check_dims_2, check_dims_3);
}

// From types.h in TensorFlow Lite
template <int N> bool IsPackedWithoutStrides(const Dims<N> &dims)
{
  int expected_stride = 1;
  for (int d = 0; d < N; d++)
  {
    if (dims.strides[d] != expected_stride)
      return false;
    expected_stride *= dims.sizes[d];
  }
  return true;
}

#endif // __DIMS_H__
