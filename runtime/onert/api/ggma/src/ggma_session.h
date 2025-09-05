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

#ifndef __API_GGMA_SESSION_H__
#define __API_GGMA_SESSION_H__

#include "ggma.h"
#include "ggma_config.h"
#include "ggma_kv_cache.h"
#include "ggma_pkg.h"
#include "nnfw.h"

#include <memory>
#include <cstdint>
#include <vector>

struct ggma_session
{
public:
  /**
   * @brief Factory method. It creates and initialize ggma_session
   *
   * @note  Use factory instead of constructor to get status
   */
  static GGMA_STATUS from_package(ggma_session **session, ggma_pkg *pkg);

private:
  ggma_session(ggma_pkg *pkg);

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
  ~ggma_session() = default;

  GGMA_STATUS generate(ggma_token *tokens, size_t n_tokens, size_t n_tokens_max, size_t *n_predict);

private:
  std::unique_ptr<ggma_pkg> _pkg;
  ggma::GGMAConfig _cfg;
  ggma::KVCache _cache;
};

#endif // __API_GGMA_SESSION_H__
