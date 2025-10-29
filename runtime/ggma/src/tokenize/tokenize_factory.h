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

#ifndef __GGMA_TOKENIZE_FACTORY_H__
#define __GGMA_TOKENIZE_FACTORY_H__

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

namespace ggma
{

class Tokenizer;

class TokenizerFactory
{
public:
  std::unordered_map<std::string, std::function<std::unique_ptr<Tokenizer>(const std::string &)>>
    _ctors;
  std::unordered_map<std::string, std::unique_ptr<Tokenizer>> _instances; // cache for instances

  static TokenizerFactory &getInstance();
  Tokenizer *create(const std::string &id, const std::string &package_path);
  void destroy(const std::string &id);
};

} // namespace ggma

#endif // __GGMA_TOKENIZE_FACTORY_H__
