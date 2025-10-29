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

#include <fstream>
#include <json/json.h>
#include <sstream>

namespace ggma
{

// Helper functions for config loading with template specialization
template <typename T>
void load_config_field(const Json::Value &root, const std::string &field_name, T &target,
                       bool is_optional = false)
{
  if (root.isMember(field_name))
    target = root[field_name].asInt();
  else if (!is_optional)
    throw std::runtime_error(field_name + " not found in config.json");
}

// Template specialization for bool type
template <>
void load_config_field<bool>(const Json::Value &root, const std::string &field_name, bool &target,
                             bool is_optional)
{
  if (root.isMember(field_name))
    target = root[field_name].asBool();
  else if (!is_optional)
    throw std::runtime_error(field_name + " not found in config.json");
}

// Template specialization for std::optional<int> type
template <>
void load_config_field<std::optional<int>>(const Json::Value &root, const std::string &field_name,
                                           std::optional<int> &target, bool is_optional)
{
  if (root.isMember(field_name))
    target = root[field_name].asInt();
}

// Constructor with default values
ModelConfig::ModelConfig() {}

// Load configuration from JSON file
void ModelConfig::load_from_file(const std::string &config_path)
{
  std::ifstream config_file(config_path);

  if (!config_file.is_open())
    throw std::runtime_error(
      "config.json not found in package. This file is required for ggma_context.");

  try
  {
    Json::Value root;
    Json::Reader reader;

    if (!reader.parse(config_file, root, false))
      throw std::runtime_error("Failed to parse JSON: " + reader.getFormattedErrorMessages());

    load_from_json(root);
  }
  catch (const std::exception &e)
  {
    // Re-throw the exception to ensure session creation fails
    throw std::runtime_error("Failed to load config.json: " + std::string(e.what()));
  }
}

// Load configuration from JSON value
void ModelConfig::load_from_json(const Json::Value &root)
{
  // Load model configuration from Hugging Face config.json
  load_config_field(root, "num_hidden_layers", n_layers);
  load_config_field(root, "hidden_size", hidden_size);
  load_config_field(root, "num_attention_heads", num_attention_heads);
  load_config_field(root, "vocab_size", vocab_size);
  load_config_field(root, "max_position_embeddings", max_position_embeddings);
  load_config_field(root, "bos_token_id", bos_token_id);
  load_config_field(root, "eos_token_id", eos_token_id);
}

// Validate configuration
bool ModelConfig::is_valid() const
{
  // Check required fields are positive
  if (n_layers <= 0)
    return false;
  if (hidden_size <= 0)
    return false;
  if (num_attention_heads <= 0)
    return false;
  if (vocab_size <= 0)
    return false;
  if (max_position_embeddings <= 0)
    return false;

  // Check token IDs are non-negative (only if they have values)
  if (bos_token_id.has_value() && bos_token_id.value() < 0)
    return false;
  if (eos_token_id.has_value() && eos_token_id.value() < 0)
    return false;

  return true;
}

// Get configuration as string (for debugging)
std::string ModelConfig::to_string() const
{
  std::ostringstream oss;
  oss << "ModelConfig {\n";
  oss << "  n_layers: " << n_layers << "\n";
  oss << "  hidden_size: " << hidden_size << "\n";
  oss << "  num_attention_heads: " << num_attention_heads << "\n";
  oss << "  vocab_size: " << vocab_size << "\n";
  oss << "  max_position_embeddings: " << max_position_embeddings << "\n";
  oss << "  bos_token_id: "
      << (bos_token_id.has_value() ? std::to_string(bos_token_id.value()) : "undefined") << "\n";
  oss << "  eos_token_id: "
      << (eos_token_id.has_value() ? std::to_string(eos_token_id.value()) : "undefined") << "\n";
  oss << "}";
  return oss.str();
}

// Utility functions for ModelConfig
bool validate_model_config(const ModelConfig &config) { return config.is_valid(); }

std::string to_string(const ModelConfig &config) { return config.to_string(); }

} // namespace ggma
