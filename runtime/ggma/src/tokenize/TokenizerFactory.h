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

#ifndef __GGMA_TOKENIZE_TOKENIZER_FACTORY_H__
#define __GGMA_TOKENIZE_TOKENIZER_FACTORY_H__

#include "Tokenizer.h"

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

namespace ggma
{
class TokenizerFactory
{
public:
  using Creator = std::function<Tokenizer *(const std::string &)>;

private:
  TokenizerFactory() = default;
  ~TokenizerFactory() = default;
  TokenizerFactory(const TokenizerFactory &) = delete;
  TokenizerFactory &operator=(const TokenizerFactory &) = delete;

  std::unordered_map<std::string, Creator> _ctors;
  static TokenizerFactory &getInstance();

public:
  static Tokenizer *create(const std::string &id, const std::string &tokenizer_dir);
  static void add(const std::string &name, const Creator &ctor);
};

} // namespace ggma

// Macro for automatic tokenizer registration
#define REGISTER_TOKENIZER(name, type)                                                       \
  namespace ggma                                                                             \
  {                                                                                          \
  struct Registrar_##type                                                                    \
  {                                                                                          \
    Registrar_##type()                                                                       \
    {                                                                                        \
      ::ggma::TokenizerFactory::add(                                                         \
        name, [](const std::string &tokenizer_dir) { return type::create(tokenizer_dir); }); \
    }                                                                                        \
  };                                                                                         \
  static Registrar_##type registrar_##type;                                                  \
  }

#endif // __GGMA_TOKENIZE_TOKENIZER_FACTORY_H__
