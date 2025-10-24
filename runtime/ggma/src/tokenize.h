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

/**
 * @file  ggma_tokenize_internal.h
 * @brief This file defines the internal GGMA Tokenizer class (C++ only).
 */
#ifndef __GGMA_TOKENIZE_INTERNAL_H__
#define __GGMA_TOKENIZE_INTERNAL_H__

#include "ggma_types.h"

namespace ggma
{

/**
 * @brief Tokenizer class for GGMA models
 */
class GGMATokenizer
{
public:
  GGMATokenizer() = default;
  ~GGMATokenizer() = default;

  // TODO: Implement tokenization methods
  size_t tokenize(const char *text, size_t text_len, int64_t *tokens, size_t max_tokens,
                  size_t *actual_count) const;
};

} // namespace ggma

#endif // __GGMA_TOKENIZE_INTERNAL_H__
