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

#ifndef __GGMA_CONTEXT_H__
#define __GGMA_CONTEXT_H__

#include "ggma_types.h"
#include "Config.h"
#include "KVCache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggma
{

class Context
{
public:
  Context(const char *package_path);
  GGMAConfig load_config(const std::string &package_path);

  void prefill(ggma_token *tokens, size_t n_tokens, std::vector<uint8_t> &hidden_state);
  void unemb(std::vector<uint8_t> &hidden_state, size_t n_tokens, std::vector<float> &logits);
  ggma_token sample(const std::vector<float> &logits);
  void decode(ggma_token token_id, std::vector<uint8_t> &hidden_state);
  void decode(ggma_token token_id, std::vector<float> &logits);

private:
  // Template implementation to eliminate code duplication
  template <bool ReturnLogits, typename OutputType>
  void decode_impl(ggma_token token_id, OutputType &output);
  void init_kv_cache();

public:
  ~Context() = default;

  GGMA_STATUS generate(ggma_token *tokens, size_t n_tokens, size_t n_tokens_max, size_t *n_predict);

private:
  std::string _package_path;
  ggma::GGMAConfig _cfg;
  ggma::KVCache _cache;
};

} // namespace ggma

#endif // __GGMA_CONTEXT_H__
