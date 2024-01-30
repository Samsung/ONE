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

// codes under namespace sparsity referenced from
// https://github.com/tensorflow/tensorflow/blob/3f878cff5b698b82eea85db2b60d65a2e320850e/
//       tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h
//       tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc

#include "SparsityFormatConverter.h"

#include <oops/InternalExn.h>

#include <cassert>

namespace sparsity
{

namespace
{

uint64_t GetFlattenedIndex(const std::vector<int> &indices, const std::vector<int> &shape)
{
  uint64_t index = 0;
  int sub_elements = 1;
  for (int i = shape.size() - 1; i >= 0; i--)
  {
    assert(indices[i] >= 0);
    assert(sub_elements >= 0);
    index += static_cast<uint64_t>(indices[i]) * static_cast<uint64_t>(sub_elements);
    sub_elements *= shape[i];
  }
  return index;
}

std::vector<int> TfLiteIntArrayToVector(const TfLiteIntArray *int_array)
{
  std::vector<int> values;
  if (!int_array)
  {
    return values;
  }

  values.resize(int_array->size);
  for (int i = 0; i < int_array->size; i++)
  {
    values[i] = int_array->data[i];
  }

  return values;
}

} // namespace

template <typename T>
FormatConverter<T>::FormatConverter(const std::vector<int> &shape, const TfLiteSparsity &sparsity)
{
  auto traversal_order = TfLiteIntArrayToVector(sparsity.traversal_order);
  auto block_map = TfLiteIntArrayToVector(sparsity.block_map);

  std::vector<TfLiteDimensionType> format(sparsity.dim_metadata_size);
  std::vector<int> dense_size(sparsity.dim_metadata_size);
  std::vector<std::vector<int>> segments(sparsity.dim_metadata_size);
  std::vector<std::vector<int>> indices(sparsity.dim_metadata_size);
  for (int i = 0; i < sparsity.dim_metadata_size; i++)
  {
    format[i] = sparsity.dim_metadata[i].format;
    dense_size[i] = sparsity.dim_metadata[i].dense_size;
    segments[i] = TfLiteIntArrayToVector(sparsity.dim_metadata[i].array_segments);
    indices[i] = TfLiteIntArrayToVector(sparsity.dim_metadata[i].array_indices);
  }

  InitSparseToDenseConverter(shape, std::move(traversal_order), std::move(format),
                             std::move(dense_size), std::move(segments), std::move(indices),
                             std::move(block_map));
}

template <typename T>
void FormatConverter<T>::InitSparseToDenseConverter(
  std::vector<int> shape, std::vector<int> traversal_order, std::vector<TfLiteDimensionType> format,
  std::vector<int> dense_size, std::vector<std::vector<int>> segments,
  std::vector<std::vector<int>> indices, std::vector<int> block_map)
{
  dense_shape_ = std::move(shape);
  traversal_order_ = std::move(traversal_order);
  block_map_ = std::move(block_map);
  format_ = std::move(format);

  dense_size_ = 1;
  for (size_t i = 0; i < dense_shape_.size(); i++)
  {
    dense_size_ *= dense_shape_[i];
  }

  dim_metadata_.resize(2 * format_.size());
  for (size_t i = 0; i < format_.size(); i++)
  {
    if (format_[i] == kTfLiteDimDense)
    {
      dim_metadata_[2 * i] = {dense_size[i]};
    }
    else
    {
      dim_metadata_[2 * i] = std::move(segments[i]);
      dim_metadata_[2 * i + 1] = std::move(indices[i]);
    }
  }

  int original_rank = dense_shape_.size();
  int block_dim = 0;

  blocked_shape_.resize(original_rank);
  block_size_.resize(block_map_.size());
  for (int i = 0; i < original_rank; i++)
  {
    if (block_dim < (int)block_map_.size() && block_map_[block_dim] == i)
    {
      if (original_rank + block_dim < (int)traversal_order_.size())
      {
        int orig_dim = traversal_order_[original_rank + block_dim];
        block_size_[block_dim] = dense_size[orig_dim];
        blocked_shape_[i] = dense_shape_[i] / dense_size[orig_dim];
        block_dim++;
      }
    }
    else
    {
      blocked_shape_[i] = dense_shape_[i];
    }
  }
}

template <typename T>
void FormatConverter<T>::Populate(const T *src_data, std::vector<int> indices, int level,
                                  int prev_idx, int *src_data_ptr, T *dest_data)
{
  if (static_cast<size_t>(level) == indices.size())
  {
    int orig_rank = dense_shape_.size();
    std::vector<int> orig_idx;
    orig_idx.resize(orig_rank);
    int i = 0;
    for (; static_cast<size_t>(i) < orig_idx.size(); i++)
    {
      int orig_dim = traversal_order_[i];
      orig_idx[orig_dim] = indices[i];
    }

    for (; static_cast<size_t>(i) < indices.size(); i++)
    {
      const int block_idx = traversal_order_[i] - orig_rank;
      const int orig_dim = block_map_[block_idx];
      orig_idx[orig_dim] = orig_idx[orig_dim] * block_size_[block_idx] + indices[i];
    }

    dest_data[GetFlattenedIndex(orig_idx, dense_shape_)] = src_data[*src_data_ptr];

    *src_data_ptr = *src_data_ptr + 1;
    return;
  }

  const int metadata_idx = 2 * level;
  const int shape_of_level = dim_metadata_[metadata_idx][0];
  if (format_[level] == kTfLiteDimDense)
  {
    for (int i = 0; i < shape_of_level; i++)
    {
      indices[level] = i;
      Populate(src_data, indices, level + 1, prev_idx * shape_of_level + i, src_data_ptr,
               dest_data);
    }
  }
  else if (static_cast<size_t>(prev_idx + 1) < dim_metadata_[metadata_idx].size())
  {
    const auto &array_segments = dim_metadata_[metadata_idx];
    const auto &array_indices = dim_metadata_[metadata_idx + 1];
    for (int i = array_segments[prev_idx]; i < array_segments[prev_idx + 1]; i++)
    {
      if (static_cast<size_t>(i) < array_indices.size() &&
          static_cast<size_t>(level) < indices.size())
      {
        indices[level] = array_indices[i];
        Populate(src_data, indices, level + 1, i, src_data_ptr, dest_data);
      }
    }
  }
}

template <typename T> bool FormatConverter<T>::SparseToDense(const T *src_data)
{
  data_.resize(dense_size_);
  std::fill(data_.begin(), data_.end(), T(0));

  int total_rank = traversal_order_.size();
  int src_data_ptr = 0;
  std::vector<int> indices(total_rank);
  Populate(src_data, indices, 0, 0, &src_data_ptr, data_.data());

  return true;
}

template class FormatConverter<float>;
template class FormatConverter<uint16_t>;

} // namespace sparsity

#include <luci/IR/SparsityParam.h>

namespace luci
{

sparsity::TfLiteDimensionType to_tflite_sparsity(luci::DimensionType dt)
{
  switch (dt)
  {
    case luci::DimensionType::DENSE:
      return sparsity::TfLiteDimensionType::kTfLiteDimDense;
    case luci::DimensionType::SPARSE_CSR:
      return sparsity::TfLiteDimensionType::kTfLiteDimSparseCSR;
  }
  return sparsity::TfLiteDimensionType::kTfLiteDimDense;
}

sparsity::TfLiteIntArray *to_tflite_sparsity(const luci::SparseIndexVector &data)
{
  auto type = data.type();
  switch (type)
  {
    case luci::SparseIndexVectorType::NONE:
    {
      std::vector<int32_t> empty;
      return makeTfLiteArray(empty);
    }
    case luci::SparseIndexVectorType::I32:
      return makeTfLiteArray<int32_t>(*data.as_int32_vector());
    case luci::SparseIndexVectorType::U16:
      return makeTfLiteArray<uint16_t>(*data.as_uint16_vector());
    case luci::SparseIndexVectorType::U8:
      return makeTfLiteArray<uint8_t>(*data.as_uint8_vector());
    default:
      INTERNAL_EXN_V("unsupported SparseIndexVectorType", oops::to_uint32(type));
  }
}

sparsity::TfLiteSparsity to_tflite_sparsity(const luci::SparsityParam *sp)
{
  sparsity::TfLiteSparsity tflsp;
  tflsp.traversal_order = makeTfLiteArray(sp->traversal_order);
  tflsp.block_map = makeTfLiteArray(sp->block_map);
  tflsp.dim_metadata = makeTfLiteDimensionMetadata(sp->dim_metadata);
  tflsp.dim_metadata_size = sp->dim_metadata.size();
  return tflsp;
}

template <typename T> sparsity::TfLiteIntArray *makeTfLiteArray(const std::vector<T> &data)
{
  size_t cn = data.size();
  size_t sz = 1 + data.size();
  sparsity::TfLiteIntArray *sp = (sparsity::TfLiteIntArray *)(new int[sz]);
  sp->size = cn;
  for (size_t i = 0; i < cn; ++i)
  {
    sp->data[i] = data[i];
  }
  return sp;
}

sparsity::TfLiteDimensionMetadata *
makeTfLiteDimensionMetadata(const std::vector<luci::DimMetaData> &data)
{
  size_t cn = data.size();
  sparsity::TfLiteDimensionMetadata *tfldm = new sparsity::TfLiteDimensionMetadata[cn];

  for (size_t i = 0; i < cn; ++i)
  {
    tfldm[i].format = to_tflite_sparsity(data[i].format());
    tfldm[i].dense_size = data[i].dense_size();
    tfldm[i].array_segments = to_tflite_sparsity(data[i].array_segments());
    tfldm[i].array_indices = to_tflite_sparsity(data[i].array_indices());
  }

  return tfldm;
}

void freeTfLiteSparsity(sparsity::TfLiteSparsity &tflsp)
{
  assert(tflsp.traversal_order);
  assert(tflsp.block_map);
  delete[] tflsp.traversal_order;
  delete[] tflsp.block_map;

  for (int i = 0; i < tflsp.dim_metadata_size; ++i)
  {
    assert(tflsp.dim_metadata[i].array_segments);
    assert(tflsp.dim_metadata[i].array_indices);
    delete[] tflsp.dim_metadata[i].array_segments;
    delete[] tflsp.dim_metadata[i].array_indices;
  }
}

} // namespace luci
