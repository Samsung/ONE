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

#ifndef __GGMA_TOKENIZE_H__
#define __GGMA_TOKENIZE_H__

#include <string>
#include <memory>
#include <functional>
#include <map>
#include "tokenize_factory.h"

namespace ggma
{

class Tokenizer
{
public:
  std::string id; // tokenizer identifier
  virtual ~Tokenizer() = default;
  virtual size_t tokenize(const char *text, size_t text_len, int32_t *tokens, size_t max_tokens,
                          size_t *n_tokens) const = 0;
  virtual size_t detokenize(const int32_t *tokens, size_t n_tokens, char *text,
                            size_t text_len) const = 0;
};

// Macro for automatic tokenizer registration
#define REGISTER_TOKENIZER(name, type)                                             \
  namespace                                                                        \
  {                                                                                \
  struct AutoRegister_##type                                                       \
  {                                                                                \
    AutoRegister_##type()                                                          \
    {                                                                              \
      ::ggma::TokenizerFactory::getInstance()._ctors[name] = ::ggma::type::create; \
    }                                                                              \
  };                                                                               \
  static AutoRegister_##type auto_register_##type;                                 \
  }

} // namespace ggma

#endif // __GGMA_TOKENIZE_H__
