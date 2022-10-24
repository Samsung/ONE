/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PASS_HELPERS_SPARSITY_FORMAT_CONVERTER_H__
#define __LUCI_PASS_HELPERS_SPARSITY_FORMAT_CONVERTER_H__

#include <cstddef>
#include <cstdint>
#include <vector>

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

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct TfLiteIntArray
{
  int size;
  int data[];
} TfLiteIntArray;

// Metadata to encode each dimension in a sparse tensor.
typedef struct TfLiteDimensionMetadata
{
  TfLiteDimensionType format;
  int dense_size;
  TfLiteIntArray *array_segments;
  TfLiteIntArray *array_indices;
} TfLiteDimensionMetadata;

// Parameters used to encode a sparse tensor. For detailed explanation of each
// field please refer to lite/schema/schema.fbs.
typedef struct TfLiteSparsity
{
  TfLiteIntArray *traversal_order;
  TfLiteIntArray *block_map;
  TfLiteDimensionMetadata *dim_metadata;
  int dim_metadata_size;
} TfLiteSparsity;

// A converter that keeps an internal representation of sparse tensor parameters
// and converts tensors between dense and sparse formats.
template <typename T> class FormatConverter
{
public:
  /* Creates a sparse to dense converter.
   * @param shape      Shape of the target dense tensor.
   * @param sparsity   Sparsity parameter of the sparse TfLiteTensor.
   */
  FormatConverter(const std::vector<int> &shape, const TfLiteSparsity &sparsity);

  const std::vector<T> &GetData() { return data_; }
  const std::vector<std::vector<int>> &GetDimMetadata() { return dim_metadata_; }

  bool SparseToDense(const T *src_data);

private:
  // Helper function for initializing this converter for sparse to dense
  // conversion.
  void InitSparseToDenseConverter(std::vector<int> shape, std::vector<int> traversal_order,
                                  std::vector<TfLiteDimensionType> format,
                                  std::vector<int> dense_size,
                                  std::vector<std::vector<int>> segments,
                                  std::vector<std::vector<int>> indices,
                                  std::vector<int> block_map);

  void Populate(const T *src_data, std::vector<int> indices, int level, int prev_idx,
                int *src_data_ptr, T *dest_data);

private:
  std::vector<int> dense_shape_;
  std::vector<int> blocked_shape_;
  size_t dense_size_;
  std::vector<int> traversal_order_;
  std::vector<TfLiteDimensionType> format_;
  std::vector<int> block_size_;
  std::vector<int> block_map_;
  std::vector<std::vector<int>> dim_metadata_;
  std::vector<T> data_;
};

extern template class FormatConverter<float>;
extern template class FormatConverter<uint16_t>;

} // namespace sparsity

#include <luci/IR/SparsityParam.h>

namespace luci
{

sparsity::TfLiteDimensionType to_tflite_sparsity(luci::DimensionType dt);
sparsity::TfLiteIntArray *to_tflite_sparsity(const luci::SparseIndexVector &data);
sparsity::TfLiteSparsity to_tflite_sparsity(const luci::SparsityParam *sp);

template <typename T> sparsity::TfLiteIntArray *makeTfLiteArray(const std::vector<T> &data);
sparsity::TfLiteDimensionMetadata *
makeTfLiteDimensionMetadata(const std::vector<luci::DimMetaData> &data);

void freeTfLiteSparsity(sparsity::TfLiteSparsity &tflsp);

} // namespace luci

#endif // __LUCI_PASS_HELPERS_SPARSITY_FORMAT_CONVERTER_H__
