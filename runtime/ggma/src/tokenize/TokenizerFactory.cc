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

#include "TokenizerFactory.h"
#include "Tokenizer.h"

namespace ggma
{

TokenizerFactory &TokenizerFactory::getInstance()
{
  static TokenizerFactory instance;
  return instance;
}

Tokenizer *TokenizerFactory::create(const std::string &id, const std::string &package_path)
{
  auto &factory = getInstance();
  auto ctor_it = factory._ctors.find(id);
  if (ctor_it != factory._ctors.end())
  {
    auto tokenizer = ctor_it->second(package_path);
    if (tokenizer)
    {
      return tokenizer;
    }
  }
  return nullptr;
}

void TokenizerFactory::add(const std::string &name, const Creator &ctor)
{
  auto &factory = getInstance();
  factory._ctors[name] = ctor;
}

} // namespace ggma
