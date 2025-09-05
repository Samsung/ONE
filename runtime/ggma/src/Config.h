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

#ifndef __GGMA_CONFIG_H__
#define __GGMA_CONFIG_H__

#include "KVCache.h"

#include <optional>
#include <string>

// Forward declaration for JSON
namespace Json
{
class Value;
}

namespace ggma
{

// Structure to hold model architecture specific configuration.
// This can be loaded from various model sources (e.g., Hugging Face, FairSeq).
struct ModelConfig
{
  //--- ModelConfig
  int n_layers;    // Number of transformer layers
  int hidden_size; // Hidden dimension
  int num_attention_heads;
  int vocab_size;              // Vocabulary size
  int max_position_embeddings; // Maximum sequence length
  std::optional<int> bos_token_id;
  std::optional<int> eos_token_id;

  // Constructor with default values
  ModelConfig();

  // Load configuration from JSON file
  void load_from_file(const std::string &config_path);

  // Load configuration from JSON value
  void load_from_json(const Json::Value &root);

  // Validate configuration
  bool is_valid() const;

  // Get configuration as string (for debugging)
  std::string to_string() const;
};

// Structure to hold all GGMA-specific configurations,
// including both the model architecture and runtime/execution parameters.
struct GGMAConfig
{
  ModelConfig model; // Model architecture details
  int cache_size = 32;
  int ubatch = 32;
  KVCacheDataType kv_cache_type = KVCacheDataType::FLOAT32; // KV cache data type
};

// Utility functions for ModelConfig
bool validate_model_config(const ModelConfig &config);
std::string to_string(const ModelConfig &config);

} // namespace ggma

#endif // __GGMA_CONFIG_H__
