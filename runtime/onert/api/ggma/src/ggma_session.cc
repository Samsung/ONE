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

#include "ggma_config.h"
#include "ggma_kv_cache.h"
#include "ggma_session.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// NNFW_ENSURE_STATUS macro
#define NNFW_ENSURE_STATUS(a)                            \
  do                                                     \
  {                                                      \
    if ((a) != NNFW_STATUS_NO_ERROR)                     \
    {                                                    \
      throw std::runtime_error("NNFW operation failed"); \
    }                                                    \
  } while (0)

// Helper function to create and prepare a model session
nnfw_session *create_and_prepare_session(const std::string &model_path)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_STATUS(nnfw_create_session(&session));
  NNFW_ENSURE_STATUS(nnfw_load_model_from_file(session, model_path.c_str()));
  NNFW_ENSURE_STATUS(nnfw_prepare(session));
  return session;
}

// Helper functions for tensor size calculation
uint64_t num_elems(const nnfw_tensorinfo *tensor_info)
{
  uint64_t n = 1;
  for (int32_t i = 0; i < tensor_info->rank; ++i)
    n *= tensor_info->dims[i];
  return n;
}

uint64_t bufsize_for(const nnfw_tensorinfo *ti)
{
  static int elmsize[] = {
    sizeof(float),   /* NNFW_TYPE_TENSOR_FLOAT32 */
    sizeof(int),     /* NNFW_TYPE_TENSOR_INT32 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_QUANT8_ASYMM */
    sizeof(bool),    /* NNFW_TYPE_TENSOR_BOOL = 3 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_UINT8 = 4 */
    sizeof(int64_t), /* NNFW_TYPE_TENSOR_INT64 = 5 */
    sizeof(int8_t),  /* NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED = 6 */
    sizeof(int16_t), /* NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED = 7 */
  };
  return elmsize[ti->dtype] * num_elems(ti);
}

ggma_session::ggma_session(ggma_pkg *pkg) : _pkg(pkg)
{
  _cfg = _pkg->load_config();
  _cache.init(_cfg, _cfg.cache_size);
}

GGMA_STATUS ggma_session::from_package(ggma_session **session, ggma_pkg *pkg)
{
  if (session == nullptr)
    return GGMA_STATUS_UNEXPECTED_NULL;
  try
  {
    auto new_session = std::unique_ptr<ggma_session>(new ggma_session(pkg));
    *session = new_session.release();
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during session creation" << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return GGMA_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during session initialization : " << e.what() << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return GGMA_STATUS_ERROR;
  }
  return GGMA_STATUS_NO_ERROR;
}

void ggma_session::prefill(ggma_token *tokens, size_t n_tokens, std::vector<uint8_t> &hidden_state)
{
  std::filesystem::path nnpkg_path = std::filesystem::path(_pkg->path()) / "prefill";
  nnfw_session *session = create_and_prepare_session(nnpkg_path.string());

  nnfw_tensorinfo ti;

  // Input 0: token_id
  //   shape = [n_batch, n_seq]
  //   n_batch = 1
  NNFW_ENSURE_STATUS(nnfw_input_tensorinfo(session, 0, &ti));
  if (ti.rank != 2 || ti.dims[0] != 1)
    throw std::runtime_error("prefill : invalid input shape");

  // TODO: Check ubatch from model is same to runtime config
  int ubatch = ti.dims[1]; // Number of tokens after padding to align to 32 multiples
  // Use tokens as input without copying (zero-copy)
  NNFW_ENSURE_STATUS(nnfw_set_input(session, 0, ti.dtype, tokens, ubatch * sizeof(ggma_token)));

  // Expected Output:
  //
  //  Index |   Name   |   Description
  //  ------|----------|---------------------------
  //   0    |  k0      |   key cache for layer 0
  //   1    |  v0      | value cache for layer 0
  //   ...  |  ...     |     ...
  //  2n-2  |  k{n-1}  |   key cache for layer n-1
  //  2n-1  |  v{n-1}  | value cache for layer n-1
  //   2n   |  hidden  |        hidden state
  //
  // where n = number of layers

  uint32_t num_outputs;
  NNFW_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
  if (num_outputs != _cfg.model.n_layers * 2 + 1)
    throw std::runtime_error("prefill : number of outputs mismatch");

  // Output 0~2n-1: KV caches
  for (int i = 0; i < _cfg.model.n_layers; ++i)
  {
    if (!_cache.v[i].empty())
      NNFW_ENSURE_STATUS(nnfw_set_output(session, 2 * i, _cache.to_nnfw_type(), _cache.v[i].data(),
                                         _cache.v[i].size()));
    if (!_cache.k[i].empty())
      NNFW_ENSURE_STATUS(nnfw_set_output(session, 2 * i + 1, _cache.to_nnfw_type(),
                                         _cache.k[i].data(), _cache.k[i].size()));
  }

  // Output 2n: hidden_state
  //   shape = [n_batch, n_seq, n_emb]
  //   n_batch = 1

  NNFW_ENSURE_STATUS(nnfw_output_tensorinfo(session, 2 * _cfg.model.n_layers, &ti));
  if (ti.rank != 3 || ti.dims[0] != 1)
    throw std::runtime_error("prefill : invalid hidden shape");

  // Allocate output buffer
  hidden_state.resize(bufsize_for(&ti), 0);
  // Output buffer setup - use externally allocated hidden_state (single output for single token)
  NNFW_ENSURE_STATUS(
    nnfw_set_output(session, num_outputs - 1, ti.dtype, hidden_state.data(), hidden_state.size()));

  NNFW_ENSURE_STATUS(nnfw_run(session));
  nnfw_close_session(session);
}

void ggma_session::unemb(std::vector<uint8_t> &hidden_state, size_t n_tokens,
                         std::vector<float> &logits)
{
  std::filesystem::path nnpkg_path = std::filesystem::path(_pkg->path()) / "unemb";
  nnfw_session *session = create_and_prepare_session(nnpkg_path.string());

  // Input buffer setup - use externally allocated hidden_state
  nnfw_tensorinfo ti;
  NNFW_ENSURE_STATUS(nnfw_input_tensorinfo(session, 0, &ti));
  // ti[0] : n_batch
  // ti[1] : n_seq = ubatch   if padded
  //               = n_tokens if not padded
  if (ti.rank != 3 || ti.dims[0] != 1)
    throw std::runtime_error("unemb : invalid input shape");
  assert(ti.dims[1] == _cfg.ubatch); // Previously, it was padded to ubatch.
  // Handle effective (actual) tokens only.
  ti.dims[1] = n_tokens;
  // Update buffer and nnfw input tensor info as sequence length is adjusted.
  hidden_state.resize(bufsize_for(&ti), 0);
  NNFW_ENSURE_STATUS(nnfw_set_input_tensorinfo(session, 0, &ti));
  NNFW_ENSURE_STATUS(
    nnfw_set_input(session, 0, ti.dtype, hidden_state.data(), hidden_state.size()));

  // Output buffer setup - use externally allocated logits
  NNFW_ENSURE_STATUS(nnfw_output_tensorinfo(session, 0, &ti));
  // Check if output data type is float
  if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
    throw std::runtime_error("unemb: output tensor must be float type");
  // Allocate output buffer
  // ti[0] : n_batch
  // ti[1] : n_seq = ubatch   if padded
  //               = n_tokens if not padded
  if (ti.rank != 3 || ti.dims[0] != 1)
    throw std::runtime_error("unemb : invalid output shape");
  assert(ti.dims[1] == _cfg.ubatch); // Previously, it was padded to ubatch.
  // Handle effective (actual) tokens only.
  ti.dims[1] = n_tokens;
  logits.resize(num_elems(&ti), 0);
  NNFW_ENSURE_STATUS(
    nnfw_set_output(session, 0, ti.dtype, logits.data(), logits.size() * sizeof(logits[0])));

  NNFW_ENSURE_STATUS(nnfw_run(session));
  nnfw_close_session(session);
}

// Template implementation to eliminate code duplication
template <bool ReturnLogits, typename OutputType>
void ggma_session::decode_impl(ggma_token token_id, OutputType &output)
{
  std::filesystem::path nnpkg_path = std::filesystem::path(_pkg->path()) / "decode";
  nnfw_session *session = create_and_prepare_session(nnpkg_path.string());

  // Expected Input:
  //
  //  Index |   Name    |   Description
  //  ------|-----------|------------------------
  //   0    | token_id  | input token to decode
  //   1    |   k0      | key cache for layer 0
  //   2    |   k1      | key cache for layer 1
  //   ...  |   ...     | ...
  //   n    |   k{n-1}  | key cache for layer n-1
  //   n+1  |   v0      | value cache for layer 0
  //   n+2  |   v1      | value cache for layer 1
  //   ...  |   ...     | ...
  //   2n   |   v{n-1}  | value cache for layer n-1
  //   2n+1 | cache_pos | current cache position
  //
  // where n = number of layers

  // Input 0: Token ID - use directly without copying
  nnfw_tensorinfo token_ti;
  NNFW_ENSURE_STATUS(nnfw_input_tensorinfo(session, 0, &token_ti));
  NNFW_ENSURE_STATUS(nnfw_set_input(session, 0, token_ti.dtype, &token_id, sizeof(ggma_token)));

  // Input 1~2n: KV caches - use directly without copying
  for (int i = 0; i < _cfg.model.n_layers; ++i)
  {
    NNFW_ENSURE_STATUS(nnfw_set_input(session, 1 + i, _cache.to_nnfw_type(), _cache.k[i].data(),
                                      _cache.k[i].size()));
  }
  for (int i = 0; i < _cfg.model.n_layers; ++i)
  {
    NNFW_ENSURE_STATUS(nnfw_set_input(session, 1 + _cfg.model.n_layers + i, _cache.to_nnfw_type(),
                                      _cache.v[i].data(), _cache.v[i].size()));
  }

  // Input 2n+1: Cache position - current position in KV cache
  int64_t cache_pos = _cache.pos();
  NNFW_ENSURE_STATUS(nnfw_set_input(session, 1 + 2 * _cfg.model.n_layers, NNFW_TYPE_TENSOR_INT32,
                                    &cache_pos, sizeof(cache_pos)));

  // Output buffer setup - mode dependent
  nnfw_tensorinfo ti;
  NNFW_ENSURE_STATUS(nnfw_output_tensorinfo(session, 0, &ti));
  if constexpr (ReturnLogits)
  {
    // Logits mode: float vector, type validation required
    if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
      throw std::runtime_error("decode: output tensor must be float type for logits");
    output.resize(bufsize_for(&ti) / sizeof(float), 0);
    NNFW_ENSURE_STATUS(
      nnfw_set_output(session, 0, ti.dtype, output.data(), output.size() * sizeof(float)));
  }
  else
  {
    // Hidden state mode: uint8_t vector, no type validation (preserve existing behavior)
    output.resize(bufsize_for(&ti), 0);
    NNFW_ENSURE_STATUS(nnfw_set_output(session, 0, ti.dtype, output.data(), output.size()));
  }

  NNFW_ENSURE_STATUS(nnfw_run(session));
  nnfw_close_session(session);
  _cache.advance_pos();
}

// Public interface functions - delegate to template implementation
void ggma_session::decode(ggma_token token_id, std::vector<uint8_t> &hidden_state)
{
  decode_impl<false, std::vector<uint8_t>>(token_id, hidden_state);
}

void ggma_session::decode(ggma_token token_id, std::vector<float> &logits)
{
  decode_impl<true, std::vector<float>>(token_id, logits);
}

// Template instantiation (required for template implementation in .cpp file)
template void ggma_session::decode_impl<false, std::vector<uint8_t>>(ggma_token token_id,
                                                                     std::vector<uint8_t> &output);
template void ggma_session::decode_impl<true, std::vector<float>>(ggma_token token_id,
                                                                  std::vector<float> &output);

// Sample token from logits using greedy sampling
// Input shape: [n_seq, vocab_size], sample from last token
ggma_token ggma_session::sample(const std::vector<float> &logits)
{
  if (logits.empty())
    throw std::runtime_error("Empty logits tensor");

  // Calculate total number of float elements in logits tensor
  size_t total_elements = logits.size();

  if (total_elements % _cfg.model.vocab_size != 0)
    throw std::runtime_error("Invalid sequence length in logits tensor");

  const float *last_logits = logits.data() + (total_elements - _cfg.model.vocab_size);

  // Find the token with maximum logit value from the last token's logits
  const float *max_elem_iter = std::max_element(last_logits, last_logits + _cfg.model.vocab_size);

  return std::distance(last_logits, max_elem_iter);
}

// Generate tokens using autoregressive decoding with optimized memory management
//
// Parameters:
// - tokens: Input/output token array (contains input tokens, stores generated tokens)
// - n_tokens: Number of input tokens (read-only)
// - n_tokens_max: Total capacity of tokens array (max writable space)
// - n_predict: In/out parameter for number of tokens to predict/actually predicted
//   - Input: Maximum number of tokens to generate
//   - Output: Actual number of tokens generated
//
// The function ensures no buffer overflow by checking against n_tokens_max
// and stops generation when either the requested number is reached or the array is full.
GGMA_STATUS ggma_session::generate(ggma_token *tokens, size_t n_tokens, size_t n_tokens_max,
                                   size_t *n_predict)
{
  try
  {
    _cache.reset_pos();

    std::vector<uint8_t> hidden;
    std::vector<float> logits;
    ggma_token new_token;

    prefill(tokens, n_tokens, hidden); // hidden = prefill(tokens)
    _cache.set_pos(n_tokens);
    _cache.transpose(true /* k */, "0213", _cfg.cache_size, _cfg.model.num_attention_heads,
                     _cfg.model.hidden_size / _cfg.model.num_attention_heads);
    unemb(hidden, n_tokens, logits); // logits = unemb(hidden)

    size_t n_possible = n_tokens_max - n_tokens;
    if (*n_predict > n_possible)
      *n_predict = n_possible;

    auto is_end_token = [this](ggma_token token) {
      return token == _cfg.model.eos_token_id.value_or(-1) || token == 0;
    };

    while ((_cache.pos() - n_tokens) < *n_predict)
    {
      new_token = sample(logits);
      tokens[n_tokens + (_cache.pos() - n_tokens)] = new_token;
      if (is_end_token(new_token))
        break;
      if (_cfg.share_unemb_weights)
      {
        decode(new_token, hidden); // hidden = decode(new_token)
        unemb(hidden, 1, logits);  // logits = unemb(hidden)
      }
      else
      {
        decode(new_token, logits); // logits = decode(new_token)
      }
    }
    *n_predict = _cache.pos() - n_tokens;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error in generate: " << e.what() << std::endl;
    return GGMA_STATUS_ERROR;
  }
  // expected
  //    prompt:     1, 21075,  7727,   550,   260, 12584, 31843,
  // generated : 1100, 7899, 289, 826, 351, 600, 2439, 288, 266, 3653, 31843, 1100, 7899, 289, 1261,
  // 291, 5869, 291, 1261, 31843
  return GGMA_STATUS_NO_ERROR;
}
