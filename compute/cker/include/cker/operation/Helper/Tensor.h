/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_HELPER_TENSOR_H__
#define __NNFW_CKER_HELPER_TENSOR_H__

#include "cker/Shape.h"
#include "cker/eigen/EigenSupport.h"

namespace nnfw
{
namespace cker
{
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex> struct TTypes
{
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    Tensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
    ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>> UnalignedTensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>>
    UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>, Eigen::Aligned>
    Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
    Scalar;
  typedef Eigen::TensorMap<
    Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>>
    UnalignedScalar;
  typedef Eigen::TensorMap<
    Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>>
    UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> Flat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned> Vec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>> UnalignedFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>>
    UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>> UnalignedVec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>> UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned> Matrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>
    ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>> UnalignedMatrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>>
    UnalignedConstMatrix;
};

typedef typename TTypes<float, 1>::Tensor32Bit::Index Index32;

template <typename T> struct InputTensor
{
  Shape shape;
  const T *buffer;
};

struct Tensor
{
  Shape shape;
  void *buffer;

public:
  bool copyFrom(const Tensor &other, const Shape &new_shape)
  {
    if (other.shape.FlatSize() != new_shape.FlatSize())
      return false;

    this->shape.ReplaceWith(new_shape.DimensionsCount(), new_shape.DimsData());
    this->buffer = other.buffer;

    return true;
  }

  template <typename T> T *base() const
  {
    return buffer == nullptr ? nullptr : reinterpret_cast<T *>(buffer);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(const std::vector<int32_t> &new_sizes)
  {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    for (size_t d = 0; d < NDIMS; d++)
    {
      dims[d] = new_sizes[d];
    }
    return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
  }

  template <typename T> typename TTypes<T>::Flat flat() { return shaped<T, 1>({shape.FlatSize()}); }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstTensor shaped(const std::vector<int32_t> new_sizes) const
  {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    for (size_t d = 0; d < NDIMS; d++)
    {
      dims[d] = new_sizes[d];
    }
    return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
  }

  // Create Eigen Tensor with current shape
  template <typename T, size_t NDIMS> typename TTypes<T, NDIMS>::Tensor shaped() const
  {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    for (size_t d = 0; d < NDIMS; d++)
    {
      dims[d] = shape.Dims(d);
    }
    return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
  }

  template <typename T> typename TTypes<T>::ConstFlat flat() const
  {
    return shaped<T, 1>({shape.FlatSize()});
  }

  template <typename T> typename TTypes<T>::ConstScalar scalar() const
  {
    return typename TTypes<T>::ConstScalar(base<T>());
  }

  template <typename T> typename TTypes<T>::Vec vec()
  {
    return tensor<T, 1>();
  }

  template <typename T> typename TTypes<T>::Matrix matrix()
  {
    return tensor<T, 2>();
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor()
  {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    for (size_t d = 0; d < NDIMS; d++)
    {
      dims[d] = shape.Dims(d);
    }
    return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
  }
}; // Tensor

template <typename DSizes> Eigen::DSizes<Index32, DSizes::count> To32BitDims(const DSizes &in)
{
  Eigen::DSizes<Index32, DSizes::count> out;
  for (int i = 0; i < DSizes::count; ++i)
  {
    out[i] = in[i];
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar, TensorType::NumIndices>::Tensor32Bit
To32Bit(TensorType in)
{
  typedef typename TTypes<typename TensorType::Scalar, TensorType::NumIndices>::Tensor32Bit RetType;
  return RetType(in.data(), To32BitDims(in.dimensions()));
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_HELPER_TENSOR_H__
