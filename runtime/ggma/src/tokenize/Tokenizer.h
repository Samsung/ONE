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

#ifndef __GGMA_TOKENIZE_TOKENIZER_H__
#define __GGMA_TOKENIZE_TOKENIZER_H__

#include <string>

namespace ggma
{

class Tokenizer
{
public:
  virtual ~Tokenizer() = default;
  virtual std::string id() const = 0;
  virtual size_t tokenize(const char *text, size_t text_len, int32_t *tokens, size_t max_tokens,
                          size_t *n_tokens) const = 0;
  virtual size_t detokenize(const int32_t *tokens, size_t n_tokens, char *text,
                            size_t text_len) const = 0;

protected:
  Tokenizer() = default; // Protected constructor to enforce factory pattern
  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;
};

} // namespace ggma

#endif // __GGMA_TOKENIZE_TOKENIZER_H__
