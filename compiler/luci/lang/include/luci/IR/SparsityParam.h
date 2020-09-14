/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_SPARSITYPARAM_H__
#define __LUCI_IR_SPARSITYPARAM_H__

#include <cstdint>
#include <vector>

namespace luci
{

enum DimensionType
{
  DENSE,
  SPARSE_CSR,
};

enum SparseIndexVectorType
{
  NONE,
  I32,
  U16,
  U8,
};

struct DimMetaData
{
  DimMetaData() = delete;
  DimMetaData(DimensionType format, int32_t dense_size, SparseIndexVectorType array_segments_type,
              void *array_segments, SparseIndexVectorType array_indices_type, void *array_indices)
      : _format{format}, _dense_size{dense_size}, _array_segments_type{array_segments_type},
        _array_segments{array_segments}, _array_indices_type{array_indices_type},
        _array_indices{array_indices}
  {
  }

  DimensionType _format{DimensionType::DENSE};
  int32_t _dense_size{0};
  SparseIndexVectorType _array_segments_type{SparseIndexVectorType::NONE};
  void *_array_segments{nullptr};
  SparseIndexVectorType _array_indices_type{SparseIndexVectorType::NONE};
  void *_array_indices{nullptr};
};

struct SparsityParam
{
  std::vector<int32_t> traversal_order;
  std::vector<int32_t> block_map;
  std::vector<DimMetaData> dim_metadata;
};

} // namespace luci

#endif // __LUCI_IR_SPARSITYPARAM_H__
