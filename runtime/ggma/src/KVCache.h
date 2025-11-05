/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __GGMA_KV_CACHE_H__
#define __GGMA_KV_CACHE_H__

#include "nnfw.h"

#include <cstdint>
#include <string>
#include <vector>

// Forward declaration to break circular dependency
namespace ggma
{
struct GGMAConfig;
}

namespace ggma
{

// KV Cache data type enumeration
enum class KVCacheDataType
{
  FLOAT32,
  UINT8
};

// Structure to hold Key-Value cache data
struct KVCache
{
  KVCacheDataType data_type;           // Data type for KV cache
  std::vector<std::vector<uint8_t>> k; // Key caches for each layer (raw byte storage)
  std::vector<std::vector<uint8_t>> v; // Value caches for each layer (raw byte storage)
  int64_t _pos = 0;                    // Current position in KV cache

  // Get element size in bytes based on data type
  size_t element_size() const
  {
    switch (data_type)
    {
      case KVCacheDataType::FLOAT32:
        return sizeof(float);
      case KVCacheDataType::UINT8:
        return 1;
      default:
        return sizeof(float);
    }
  }

  // Convert to NNFW tensor type
  NNFW_TYPE to_nnfw_type() const
  {
    switch (data_type)
    {
      case KVCacheDataType::FLOAT32:
        return NNFW_TYPE_TENSOR_FLOAT32;
      case KVCacheDataType::UINT8:
        return NNFW_TYPE_TENSOR_UINT8;
      default:
        return NNFW_TYPE_TENSOR_FLOAT32;
    }
  }

  // Check if KV cache is valid (properly initialized)
  bool is_valid() const
  {
    if (k.size() != v.size())
      return false;

    for (size_t i = 0; i < k.size(); ++i)
      if (k[i].size() != v[i].size())
        return false;

    return true;
  }

  // Position management methods
  int64_t pos() const { return _pos; }
  void set_pos(int pos) { _pos = pos; }
  void reset_pos() { _pos = 0; }
  void advance_pos() { _pos++; }

  // Initialize KV cache
  void init(const ggma::GGMAConfig &cfg, int cache_size);

  /**
   * @brief Transpose cache with "0213" permutation [0,2,1,3]
   * @param is_k_cache true for K cache, false for V cache
   * @param perm Permutation string (must be "0213")
   * @param seq_len Sequence length dimension
   * @param num_heads Number of attention heads
   * @param head_dim Head dimension
   */
  void transpose(bool is_k_cache, const char *perm, size_t seq_len, size_t num_heads,
                 size_t head_dim);
};

// Utility functions for KVCacheDataType
const char *to_string(KVCacheDataType type);
KVCacheDataType from_string(const std::string &type_str);
bool is_supported_type(KVCacheDataType type);

} // namespace ggma

#endif // __GGMA_KV_CACHE_H__
