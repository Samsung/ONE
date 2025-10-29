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

#include "ggma_types.h"
#include "tokenize.h"
#include "tokenize_factory.h"

#include <cstring>

extern "C" {

GGMA_STATUS ggma_create_tokenizer(ggma_tokenizer **tokenizer, const char *tokenizer_path)
{
  if (!tokenizer || !tokenizer_path)
    return GGMA_STATUS_UNEXPECTED_NULL;

  try
  {
    std::string tokenizer_id = "sentencepiece";
    auto impl = ggma::TokenizerFactory::getInstance().create(tokenizer_id, tokenizer_path);

    *tokenizer = reinterpret_cast<ggma_tokenizer *>(impl); // Factory owns the tokenizer
    return GGMA_STATUS_NO_ERROR;
  }
  catch (...)
  {
    return GGMA_STATUS_ERROR;
  }
}

GGMA_STATUS ggma_free_tokenizer(ggma_tokenizer *tokenizer)
{
  if (!tokenizer)
    return GGMA_STATUS_UNEXPECTED_NULL;

  try
  {
    auto impl = reinterpret_cast<const ggma::Tokenizer *>(tokenizer);
    ggma::TokenizerFactory::getInstance().destroy(impl->id);
    return GGMA_STATUS_NO_ERROR;
  }
  catch (...)
  {
    return GGMA_STATUS_ERROR;
  }
}

GGMA_STATUS ggma_tokenize(const ggma_tokenizer *tokenizer, const char *text, size_t text_len,
                          int32_t *tokens, size_t n_tokens_max, size_t *n_tokens)
{
  if (!tokenizer || !text || !tokens || !n_tokens)
    return GGMA_STATUS_UNEXPECTED_NULL;

  try
  {
    auto impl = reinterpret_cast<const ggma::Tokenizer *>(tokenizer);
    impl->tokenize(text, text_len, tokens, n_tokens_max, n_tokens);
    return GGMA_STATUS_NO_ERROR;
  }
  catch (...)
  {
    return GGMA_STATUS_ERROR;
  }
}

GGMA_STATUS ggma_detokenize(const ggma_tokenizer *tokenizer, const int32_t *tokens, size_t n_tokens,
                            char *text, size_t text_len)
{
  if (!tokenizer || !tokens || !text)
    return GGMA_STATUS_UNEXPECTED_NULL;

  try
  {
    auto impl = reinterpret_cast<const ggma::Tokenizer *>(tokenizer);
    impl->detokenize(tokens, n_tokens, text, text_len);
    return GGMA_STATUS_NO_ERROR;
  }
  catch (...)
  {
    return GGMA_STATUS_ERROR;
  }
}

} // extern "C"
