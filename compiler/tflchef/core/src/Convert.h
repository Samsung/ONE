/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  Convert.h
 * @brief This header declares various as_tflite_TYPE functions
 */
#ifndef __CONVERT_H__
#define __CONVERT_H__

#include <tflchef.pb.h>
#include <mio/tflite/schema_generated.h>

tflite::Padding as_tflite_padding(const tflchef::Padding &value);
tflite::ActivationFunctionType as_tflite_activation(const tflchef::Activation &value);
tflite::TensorType as_tflite_tensortype(const tflchef::TensorType &value);
tflite::MirrorPadMode as_tflite_mirrorpadmode(const tflchef::MirrorPadMode &value);
tflite::DimensionType as_tflite_dimensiontype(const tflchef::DimensionType &value);
tflite::SparseIndexVector as_tflite_sparse_idx_vec_type(const tflchef::SparseIndexVecType &value);
flatbuffers::Offset<void>
as_tflite_sparse_index_vec(flatbuffers::FlatBufferBuilder &fb,
                           const ::tflchef::TensorSparsity_IndexVec &value);

// codes under namespace sparsity referenced from
// https://github.com/tensorflow/tensorflow/blob/3f878cff5b698b82eea85db2b60d65a2e320850e/
//       tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h
//       tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc

namespace sparsity
{

// Storage format of each dimension in a sparse tensor.
typedef enum TfLiteDimensionType
{
  kTfLiteDimDense = 0,
  kTfLiteDimSparseCSR,
} TfLiteDimensionType;

template <typename T> class FormatConverter
{
public:
  FormatConverter(const std::vector<int32_t> &shape, const std::vector<int32_t> &traversal_order,
                  const std::vector<TfLiteDimensionType> &format,
                  const std::vector<int32_t> &block_size = {},
                  const std::vector<int32_t> &block_map = {});

  bool DenseToSparse(const T *src_data);

  const std::vector<T> &GetData() { return data_; }
  const std::vector<std::vector<int32_t>> &GetDimMetadata() { return dim_metadata_; }

private:
  bool IsZero(const T val);

private:
  std::vector<int32_t> dense_shape_;
  std::vector<int32_t> blocked_shape_;
  size_t dense_size_;
  std::vector<int32_t> traversal_order_;
  std::vector<TfLiteDimensionType> format_;
  std::vector<int32_t> block_size_;
  std::vector<int32_t> block_map_;
  std::vector<std::vector<int32_t>> dim_metadata_;
  std::vector<T> data_;
};

extern template class FormatConverter<float>;
extern template class FormatConverter<uint16_t>; // float16

} // namespace sparsity

#endif // __CONVERT_H__
