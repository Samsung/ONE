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
#include <stdexcept>
#include <utility>
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

class SparseIndexVector
{
public:
  SparseIndexVector() = default;
  SparseIndexVector(const SparseIndexVectorType &type, const std::vector<int32_t> &sparse_index_vec)
    : _type{type}
  {
    switch (type)
    {
      case SparseIndexVectorType::NONE:
        break;
      case SparseIndexVectorType::I32:
      {
        _vec_ptr = static_cast<void *>(
          new std::vector<int32_t>(sparse_index_vec.begin(), sparse_index_vec.end()));
        break;
      }
      case SparseIndexVectorType::U16:
      {
        auto new_vec = new std::vector<uint16_t>(sparse_index_vec.size());
        for (uint32_t idx = 0; idx < sparse_index_vec.size(); idx++)
        {
          new_vec->at(idx) = static_cast<uint16_t>(sparse_index_vec.at(idx));
        }
        _vec_ptr = static_cast<void *>(new_vec);
        break;
      }
      case SparseIndexVectorType::U8:
      {
        auto new_vec = new std::vector<uint8_t>(sparse_index_vec.size());
        for (uint32_t idx = 0; idx < sparse_index_vec.size(); idx++)
        {
          new_vec->at(idx) = static_cast<uint8_t>(sparse_index_vec.at(idx));
        }
        _vec_ptr = static_cast<void *>(new_vec);
        break;
      }
      default:
        std::runtime_error("Invalid SparseIndexVectorType");
    }
  }

  SparseIndexVector(SparseIndexVectorType type, const void *sparse_index_vec) : _type{type}
  {
    switch (type)
    {
      case SparseIndexVectorType::NONE:
        break;
      case SparseIndexVectorType::I32:
      {
        const std::vector<int32_t> *vec =
          static_cast<const std::vector<int32_t> *>(sparse_index_vec);
        _vec_ptr = static_cast<void *>(new std::vector<int32_t>(vec->begin(), vec->end()));
        break;
      }
      case SparseIndexVectorType::U16:
      {
        const std::vector<uint16_t> *vec =
          static_cast<const std::vector<uint16_t> *>(sparse_index_vec);
        _vec_ptr = static_cast<void *>(new std::vector<uint16_t>(vec->begin(), vec->end()));
        break;
      }
      case SparseIndexVectorType::U8:
      {
        const std::vector<uint8_t> *vec =
          static_cast<const std::vector<uint8_t> *>(sparse_index_vec);
        _vec_ptr = static_cast<void *>(new std::vector<uint8_t>(vec->begin(), vec->end()));
        break;
      }
      default:
        std::runtime_error("Invalid SparseIndexVectorType");
    }
  }

  SparseIndexVector(const SparseIndexVector &sparse_index_vec)
    : SparseIndexVector(sparse_index_vec._type, sparse_index_vec._vec_ptr)
  {
  }

  SparseIndexVector(SparseIndexVector &&sparse_index_vec)
    : _type{sparse_index_vec._type}, _vec_ptr{std::exchange(sparse_index_vec._vec_ptr, nullptr)}
  {
  }

  SparseIndexVector &operator=(const SparseIndexVector &sparse_index_vec)
  {
    return *this = SparseIndexVector(sparse_index_vec);
  }

  SparseIndexVector &operator=(SparseIndexVector &&sparse_index_vector)
  {
    std::swap(_vec_ptr, sparse_index_vector._vec_ptr);
    std::swap(_type, sparse_index_vector._type);
    return *this;
  }

  ~SparseIndexVector()
  {
    switch (_type)
    {
      case SparseIndexVectorType::NONE:
        break;
      case SparseIndexVectorType::I32:
      {
        auto vec_ptr = static_cast<std::vector<int32_t> *>(_vec_ptr);
        delete vec_ptr;
        break;
      }
      case SparseIndexVectorType::U16:
      {
        auto vec_ptr = static_cast<std::vector<uint16_t> *>(_vec_ptr);
        delete vec_ptr;
        break;
      }
      case SparseIndexVectorType::U8:
      {
        auto vec_ptr = static_cast<std::vector<uint8_t> *>(_vec_ptr);
        delete vec_ptr;
        break;
      }
      default:
        break;
    }
    _vec_ptr = nullptr;
    _type = SparseIndexVectorType::NONE;
  }

public:
  SparseIndexVectorType type(void) const { return _type; }

public:
  const std::vector<int32_t> *as_int32_vector(void) const
  {
    return _type == SparseIndexVectorType::I32 ? static_cast<const std::vector<int32_t> *>(_vec_ptr)
                                               : nullptr;
  }
  const std::vector<uint16_t> *as_uint16_vector(void) const
  {
    return _type == SparseIndexVectorType::U16
             ? static_cast<const std::vector<uint16_t> *>(_vec_ptr)
             : nullptr;
  }
  const std::vector<uint8_t> *as_uint8_vector(void) const
  {
    return _type == SparseIndexVectorType::U8 ? static_cast<const std::vector<uint8_t> *>(_vec_ptr)
                                              : nullptr;
  }

private:
  SparseIndexVectorType _type{SparseIndexVectorType::NONE};
  void *_vec_ptr{nullptr};
};

class DimMetaData
{
public:
  DimMetaData() = delete;
  DimMetaData(DimensionType format, int32_t dense_size) : _format{format}, _dense_size{dense_size}
  {
    // DO NOTHING
  }
  DimMetaData(DimensionType format, int32_t dense_size, const SparseIndexVector &array_segments,
              const SparseIndexVector &array_indices)
    : _format{format}, _dense_size{dense_size}, _array_segments{array_segments},
      _array_indices{array_indices}
  {
    // DO NOTHING
  }

public:
  DimensionType format(void) const { return _format; }
  int32_t dense_size(void) const { return _dense_size; }
  const SparseIndexVector &array_segments(void) const { return _array_segments; }
  const SparseIndexVector &array_indices(void) const { return _array_indices; }

private:
  DimensionType _format{DimensionType::DENSE};
  int32_t _dense_size{0};
  SparseIndexVector _array_segments;
  SparseIndexVector _array_indices;
};

struct SparsityParam
{
  std::vector<int32_t> traversal_order;
  std::vector<int32_t> block_map;
  std::vector<DimMetaData> dim_metadata;
};

} // namespace luci

#endif // __LUCI_IR_SPARSITYPARAM_H__
