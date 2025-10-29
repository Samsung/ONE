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
#include "context.h"
#include "kv_cache.h"
#include "tokenize.h"

#include <iostream>
#include <vector>

namespace ggma
{

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
GGMA_STATUS context::generate(ggma_token *tokens, size_t n_tokens, size_t n_tokens_max,
                              size_t *n_predict)
{
  try
  {
    _cache.reset_pos();

    std::vector<uint8_t> hidden;
    std::vector<float> logits;
    ggma_token new_token;

    // 1. Prefill: run the model on the initial prompt to obtain the initial hidden state.
    prefill(tokens, n_tokens, hidden); // hidden = prefill(tokens)

    // 2. Set cache position to the length of the prompt.
    _cache.set_pos(n_tokens);

    // 3. Transpose KV caches to the layout expected by the decoder.
    _cache.transpose(true /* k */, "0213", _cfg.model.num_attention_heads, _cfg.cache_size,
                     _cfg.model.hidden_size / _cfg.model.num_attention_heads);
    _cache.transpose(false /* v */, "0213", _cfg.model.num_attention_heads, _cfg.cache_size,
                     _cfg.model.hidden_size / _cfg.model.num_attention_heads);

    // 4. Unembed: obtain logits from the hidden state.
    unemb(hidden, n_tokens, logits); // logits = unemb(hidden)

    // 5. Determine how many tokens we can actually generate.
    size_t n_possible = n_tokens_max - n_tokens;
    if (*n_predict > n_possible)
      *n_predict = n_possible;

    auto is_end_token = [this](ggma_token token) {
      return token == _cfg.model.eos_token_id.value_or(-1) || token == 0;
    };

    // 6. Autoregressive generation loop.
    while ((_cache.pos() - n_tokens) < *n_predict)
    {
      // Sample the most probable token from the logits of the last position.
      new_token = sample(logits);
      tokens[n_tokens + (_cache.pos() - n_tokens)] = new_token;

      // Stop if we hit an EOS or padding token.
      if (is_end_token(new_token))
        break;

      // Decode: run the model for the newly generated token to update hidden state.
      decode(new_token, hidden); // hidden = decode(new_token)

      // Unembed: get logits for the next step.
      unemb(hidden, 1, logits); // logits = unemb(hidden)
    }

    // Report how many tokens were actually generated.
    *n_predict = _cache.pos() - n_tokens;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error in generate: " << e.what() << std::endl;
    return GGMA_STATUS_ERROR;
  }
  return GGMA_STATUS_NO_ERROR;
}

} // namespace ggma
