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

#include "config.h"
#include "kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace ggma
{

// Convert KVCacheDataType to string representation
const char *to_string(KVCacheDataType type)
{
  switch (type)
  {
    case KVCacheDataType::FLOAT32:
      return "FLOAT32";
    case KVCacheDataType::UINT8:
      return "UINT8";
    default:
      return "UNKNOWN";
  }
}

// Convert string to KVCacheDataType
KVCacheDataType from_string(const std::string &type_str)
{
  if (type_str == "FLOAT32" || type_str == "float32")
  {
    return KVCacheDataType::FLOAT32;
  }
  else if (type_str == "UINT8" || type_str == "uint8")
  {
    return KVCacheDataType::UINT8;
  }
  throw std::runtime_error("Unsupported KV cache data type: " + type_str);
}

// Check if KVCacheDataType is supported
bool is_supported_type(KVCacheDataType type)
{
  switch (type)
  {
    case KVCacheDataType::FLOAT32:
    case KVCacheDataType::UINT8:
      return true;
    default:
      return false;
  }
}

void KVCache::transpose(bool is_k_cache, const char *perm, size_t seq_len, size_t num_heads,
                        size_t head_dim)
{
  if (perm == nullptr || strcmp(perm, "0213") != 0)
    throw std::runtime_error("Only \"0213\" permutation is supported");

  std::vector<std::vector<uint8_t>> &cache_vector = is_k_cache ? k : v;
  const size_t element_bytes = element_size();
  const size_t head_bytes = head_dim * element_bytes;

  for (size_t i = 0; i < cache_vector.size(); ++i)
  {
    std::vector<uint8_t> transposed_cache = cache_vector[i];
    uint8_t *input_data = cache_vector[i].data();
    uint8_t *output_data = transposed_cache.data();

    for (size_t s = 0; s < seq_len; ++s) // seq_len
    {
      for (size_t h = 0; h < num_heads; ++h) // num_heads
      {
        // source offset: s * (num_heads * head_bytes) + h * head_bytes
        // target offset: h * (seq_len * head_bytes) + s * head_bytes
        uint8_t *src_ptr = input_data + s * (num_heads * head_bytes) + h * head_bytes;
        uint8_t *dst_ptr = output_data + h * (seq_len * head_bytes) + s * head_bytes;
        memcpy(dst_ptr, src_ptr, head_bytes);
      }
    }

    cache_vector[i] = std::move(transposed_cache);
  }
}

void KVCache::init(const ggma::GGMAConfig &cfg, int cache_size)
{
  if (cfg.model.n_layers <= 0)
    throw std::runtime_error("n_layers not properly initialized");

  // Set KV cache data type from config
  data_type = cfg.kv_cache_type;

  // Allocate space for K and V caches for each layer
  // Total: n_layers * 2 vectors (K and V for each layer)
  k.resize(cfg.model.n_layers);
  v.resize(cfg.model.n_layers);

  for (int i = 0; i < cfg.model.n_layers; ++i)
  {
    size_t buffer_size = cfg.model.hidden_size * cache_size * element_size();
    k[i].resize(buffer_size, 0);
    v[i].resize(buffer_size, 0);
  }
}

} // namespace ggma
