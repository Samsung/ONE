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

#include "ggma_api.h"
#include "context.h"
#include "package.h"
#include "tokenize.h"

#include <cstring>
#include <iostream>

// Double-check enum value changes

#define GGMA_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return GGMA_STATUS_UNEXPECTED_NULL; \
  } while (0)

GGMA_STATUS ggma_create_package(ggma_package **pkg, const char *path)
{
  GGMA_RETURN_ERROR_IF_NULL(pkg);
  try
  {
    *pkg = reinterpret_cast<ggma_package *>(new ggma::package(path));
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during ggma_pkg creation" << std::endl;
    *pkg = nullptr;
    return GGMA_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during ggma_pkg initialization : " << e.what() << std::endl;
    *pkg = nullptr;
    return GGMA_STATUS_ERROR;
  }
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_free_package(ggma_package *pkg)
{
  delete reinterpret_cast<ggma::package *>(pkg);
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_tokenize(const struct ggma_package *pkg, const char *text, size_t text_len,
                          ggma_token *tokens, size_t n_tokens_max, size_t *n_tokens)
{
  if (!pkg || !text || !tokens || !n_tokens)
  {
    return GGMA_STATUS_UNEXPECTED_NULL;
  }

  const auto *tokenizer = reinterpret_cast<const ggma::package *>(pkg)->get_tokenizer();
  if (!tokenizer)
  {
    return GGMA_STATUS_ERROR;
  }

  size_t result = tokenizer->tokenize(text, text_len, tokens, n_tokens_max, n_tokens);
  return (result > 0) ? GGMA_STATUS_NO_ERROR : GGMA_STATUS_ERROR;
}

GGMA_STATUS ggma_create_context(ggma_context **context, ggma_package *pkg)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return ggma::context::from_package(context, pkg);
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
