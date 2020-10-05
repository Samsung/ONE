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

#ifndef __LUCI_SPARSIFY_UTILS_H__
#define __LUCI_SPARSIFY_UTILS_H__

#include <vector>

#include <luci/IR/SparsityParam.h>

namespace luci
{

template <typename T> class Sparsifier
{
public:
  /*
   * Creates a dense to sparse converter.
   * @param shape             Shape of the dense tensor.
   * @param traversal_order   In what order to traverse all dimensions,
   *                          including block dimensions.
   * @param format            Whether each dimension in the dense tensor is
   *                          dense or sparse (not in the traversal order).
   * @param block_size        Size of each block dimension.
   * @param block_map         Map from block dimension to original tensor
   *                          dimension.
   */
  Sparsifier(const std::vector<int> &shape, const std::vector<int> &traversal_order,
             const std::vector<DimensionType> &format, const std::vector<int> &block_size = {},
             const std::vector<int> &block_map = {});

  std::vector<T> GetData() { return data_; }
  std::vector<std::vector<int>> GetDimMetadata() { return dim_metadata_; }

  void DenseToSparse(const T *src_data);

private:
  // Check if val is equal to zero.
  bool IsZero(const T val);

  // Shape of the conceptual dense tensor.
  std::vector<int> dense_shape_;
  // Shape of the dense tensor with inner blocks reduced. For example, a (4, 4)
  // tensor with (2, 2) block has blocked_shape (2, 2).
  std::vector<int> blocked_shape_;
  // Total number of elements in the dense tensor.
  uint64_t dense_size_;
  // Has n(original dimension)+k(block_dimension) elements.
  std::vector<int> traversal_order_;
  // Format of each dimension in the traversal order.
  std::vector<DimensionType> format_;
  // Size of each block dimension, in the same order as block map.
  std::vector<int> block_size_;
  // Map from block dimension to the original tensor dimension.
  std::vector<int> block_map_;
  // Metadata of each dimension in the traversal order.
  // Each dimension needs two vectors. For dense dimensions, the first vector
  // stores the size of that dimension, and the second vector is empty. For
  // sparse dimensions, the first vector stores the segments and the second one
  // stores the indices.
  std::vector<std::vector<int>> dim_metadata_;
  // Actual buffer holding data after conversion. Could be sparse buffer or
  // dense buffer.
  std::vector<T> data_;
};

extern template class Sparsifier<int32_t>;
extern template class Sparsifier<int8_t>;
extern template class Sparsifier<float>;

} // namespace luci

#endif // __LUCI_SPARSIFY_UTILS_H__
