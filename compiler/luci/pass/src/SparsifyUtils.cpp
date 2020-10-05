/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SparsifyUtils.h"

namespace luci
{

template <typename T>
Sparsifier<T>::Sparsifier(const std::vector<int32_t> &shape,
                          const std::vector<int32_t> &traversal_order,
                          const std::vector<DimensionType> &format,
                          const std::vector<int32_t> &block_size,
                          const std::vector<int32_t> &block_map)
    : dense_shape_(shape), traversal_order_(traversal_order), block_size_(block_size),
      block_map_(block_map)
{
  dense_size_ = 1;
  int32_t block_dim = 0;
  blocked_shape_.resize(shape.size());
  format_.resize(shape.size() + block_map.size());
  for (int32_t i = 0; i < static_cast<int32_t>(shape.size()); i++)
  {
    format_[i] = format[traversal_order[i]];
    dense_size_ *= shape[i];
    if (block_dim < static_cast<int32_t>(block_map.size()) && block_map[block_dim] == i)
    {
      blocked_shape_[i] = shape[i] / block_size[block_dim];
      block_dim++;
    }
    else
    {
      blocked_shape_[i] = shape[i];
    }
  }

  // Only dense blocks are supported.
  for (uint32_t i = 0; i < block_map.size(); i++)
  {
    format_[i + shape.size()] = DimensionType::DENSE;
  }
}

template <typename T> void Sparsifier<T>::DenseToSparse(const T *src_data)
{
  int num_original_dims = dense_shape_.size();
  int num_block_dims = block_map_.size();
  int num_expanded_dims = num_original_dims + num_block_dims;
  std::vector<int> expanded_shape(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; i++)
  {
    if (i < num_original_dims)
    {
      expanded_shape[i] = blocked_shape_[i];
    }
    else
    {
      expanded_shape[i] = block_size_[i - num_original_dims];
    }
  }

  std::vector<int> shape_offset(num_original_dims);
  shape_offset[shape_offset.size() - 1] = 1;
  for (int i = num_original_dims - 1; i > 0; --i)
  {
    shape_offset[i - 1] = shape_offset[i] * dense_shape_[i];
  }

  std::vector<int> expanded_shape_offset(num_expanded_dims);
  for (int i = 0; i < num_original_dims; ++i)
  {
    expanded_shape_offset[i] = shape_offset[i];
  }
  for (int i = 0; i < num_block_dims; ++i)
  {
    int mapped_dim = block_map_[i];
    expanded_shape_offset[num_original_dims + i] = shape_offset[mapped_dim];
    expanded_shape_offset[mapped_dim] *= block_size_[i];
  }

  std::vector<int> dst_ordered_offset(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; ++i)
  {
    dst_ordered_offset[i] = expanded_shape_offset[traversal_order_[i]];
  }

  std::vector<bool> dst_dim_has_nonzeroes(num_expanded_dims);
  std::fill(dst_dim_has_nonzeroes.begin(), dst_dim_has_nonzeroes.end(), false);
  std::vector<int> inner_compressed_dim(num_expanded_dims);
  int most_recent_compressed_dim = -1;
  std::vector<int> num_segments_of_next_compressed_dim(num_expanded_dims);
  int segment_count = 1;
  for (int i = num_expanded_dims - 1; i >= 0; --i)
  {
    inner_compressed_dim[i] = most_recent_compressed_dim;
    if (format_[i] == DimensionType::SPARSE_CSR)
    {
      most_recent_compressed_dim = i;
      num_segments_of_next_compressed_dim[i] = segment_count;
      segment_count = 1;
    }
    else
    {
      num_segments_of_next_compressed_dim[i] = -1;
      segment_count *= expanded_shape[traversal_order_[i]];
    }
  }

  dim_metadata_.resize(num_expanded_dims * 2);
  std::vector<int> dst_sparse_dims;
  dst_sparse_dims.reserve(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; ++i)
  {
    dim_metadata_[i * 2].clear();
    dim_metadata_[i * 2 + 1].clear();
    if (format_[i] == DimensionType::DENSE)
    {
      // If dimension is dense, just store the shape.
      dim_metadata_[i * 2].push_back(expanded_shape[traversal_order_[i]]);
    }
    else
    {
      dim_metadata_[i * 2].push_back(0); // Segment array always begins with 0.
      dst_sparse_dims.push_back(i);      // Add dimension to the sparse list.
    }
  }

  // This algorithm assumes that the block size is small enough for all the
  // elements to fit in cache, so the strided accesses from different traversal
  // order and the write-first-erase-later strategy shouldn't be too slow
  int dst_dim_idx = num_expanded_dims;
  std::vector<int> coordinate(num_expanded_dims, 0);
  int dense_tensor_idx = 0;
  while (dst_dim_idx >= 0)
  {
    if (dst_dim_idx == num_expanded_dims)
    {
      // We have a complete coordinate. Add the element to the value array if it
      // is not zero, or if the last dimension is dense.
      if (!IsZero(src_data[dense_tensor_idx]))
      {
        data_.push_back(src_data[dense_tensor_idx]);
        // Mark all sparse dimensions that their current indices have nonzeroes.
        for (auto dst_dim : dst_sparse_dims)
        {
          if (!dst_dim_has_nonzeroes[dst_dim])
          {
            // Only add the index to the indices array if the current nonzero
            // is the first nonzero of the block.
            dim_metadata_[2 * dst_dim + 1].push_back(coordinate[dst_dim]);
            dst_dim_has_nonzeroes[dst_dim] = true;
          }
        }
      }
      else if (format_[num_expanded_dims - 1] == DimensionType::DENSE)
      {
        data_.push_back(src_data[dense_tensor_idx]);
      }
      --dst_dim_idx;
    }
    else
    {
      int original_dim_idx = traversal_order_[dst_dim_idx];
      int dim_size = expanded_shape[original_dim_idx];
      if (dst_dim_has_nonzeroes[dst_dim_idx])
      {
        // If the previous block has nonzeroes, reset the flag to false since
        // we have just moved to a new block.
        dst_dim_has_nonzeroes[dst_dim_idx] = false;
      }
      else if (format_[dst_dim_idx] == DimensionType::SPARSE_CSR)
      {
        // This block is empty. Delete unnecessary values if compressed.
        int next_compressed_dim = inner_compressed_dim[dst_dim_idx];
        int erase_offset = dim_metadata_[2 * dst_dim_idx + 1].size() *
                           num_segments_of_next_compressed_dim[dst_dim_idx];
        if (next_compressed_dim >= 0)
        {
          auto &segments = dim_metadata_[2 * inner_compressed_dim[dst_dim_idx]];
          segments.erase(segments.begin() + 1 + erase_offset, segments.end());
        }
        else
        {
          data_.erase(data_.begin() + erase_offset, data_.end());
        }
      }
      if (++coordinate[dst_dim_idx] < dim_size)
      {
        // The current dst_dim_idx is valid (not out of bound).
        dense_tensor_idx += dst_ordered_offset[dst_dim_idx];
        ++dst_dim_idx;
      }
      else
      {
        // dst_dim_idx has reached its dim size. Update segment array and go
        // back to incrementing the previous dimension (dst_dim_idx - 1).
        if (format_[dst_dim_idx] == DimensionType::SPARSE_CSR)
        {
          dim_metadata_[2 * dst_dim_idx].push_back(dim_metadata_[2 * dst_dim_idx + 1].size());
        }
        coordinate[dst_dim_idx] = -1;
        dense_tensor_idx -= dst_ordered_offset[dst_dim_idx] * dim_size;
        --dst_dim_idx;
      }
    }
  }
}

template <typename T> bool Sparsifier<T>::IsZero(const T val) { return (val == 0); }

template class Sparsifier<int32_t>;
template class Sparsifier<int8_t>;
template class Sparsifier<float>;

} // namespace luci
