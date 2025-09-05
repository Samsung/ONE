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

#include "ggma.h"
#include "ggma_context.h"
#include "ggma_pkg.h"

#include <cstring>
#include <iostream>

// Double-check enum value changes

#define GGMA_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return GGMA_STATUS_UNEXPECTED_NULL; \
  } while (0)

GGMA_STATUS ggma_create_package(ggma_pkg **pkg, const char *path)
{
  GGMA_RETURN_ERROR_IF_NULL(pkg);
  try
  {
    *pkg = new ggma_pkg(path);
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

GGMA_STATUS ggma_free_package(ggma_pkg *pkg)
{
  delete pkg;
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_tokenize(const struct ggma_pkg *, const char *, size_t, ggma_token *tokens,
                          size_t n_tokens_max, size_t *n_tokens)
{
  // [G] TODO: it always returns tokens for "Lily picked up a flower."
  ggma_token tokenized[32] = {
    1, 21075, 7727, 550, 260, 12584, 31843,
  };
  *n_tokens = 7;
  memcpy(tokens, tokenized, sizeof(tokenized));
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_create_context(ggma_context **context, ggma_pkg *pkg)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return ggma_context::from_package(reinterpret_cast<ggma_context **>(context), pkg);
}

GGMA_STATUS ggma_free_context(ggma_context *context)
{
  delete reinterpret_cast<ggma_context *>(context);
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_generate(ggma_context *context, ggma_token *tokens, size_t n_tokens,
                          size_t n_tokens_max, size_t *n_tokens_out)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return reinterpret_cast<ggma_context *>(context)->generate(tokens, n_tokens, n_tokens_max,
                                                             n_tokens_out);
}
