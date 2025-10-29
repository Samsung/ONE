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

#include "context.h"
#include "ggma_api.h"
#include "ggma_tokenize.h"

#include <cstring>
#include <iostream>

// Double-check enum value changes

#define GGMA_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return GGMA_STATUS_UNEXPECTED_NULL; \
  } while (0)

GGMA_STATUS ggma_tokenize(const char *package_path, const char *text, size_t text_len,
                          ggma_token *tokens, size_t n_tokens_max, size_t *n_tokens)
{
  if (!package_path || !text || !tokens || !n_tokens)
  {
    return GGMA_STATUS_UNEXPECTED_NULL;
  }

  // Create tokenizer from package path
  ggma_tokenizer *tokenizer = nullptr;
  GGMA_STATUS status = ggma_create_tokenizer(&tokenizer, package_path);
  if (status != GGMA_STATUS_NO_ERROR)
  {
    return status;
  }

  // Tokenize the text using the tokenizer implementation
  status = ggma_tokenize(tokenizer, text, text_len, tokens, n_tokens_max, n_tokens);

  // Clean up
  ggma_free_tokenizer(tokenizer);

  return status;
}

GGMA_STATUS ggma_create_context(ggma_context **context, const char *package_path)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return ggma::context::from_package(context, package_path);
}

GGMA_STATUS ggma_free_context(ggma_context *context)
{
  delete reinterpret_cast<ggma::context *>(context);
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_generate(ggma_context *context, ggma_token *tokens, size_t n_tokens,
                          size_t n_tokens_max, size_t *n_tokens_out)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return reinterpret_cast<ggma::context *>(context)->generate(tokens, n_tokens, n_tokens_max,
                                                              n_tokens_out);
}
