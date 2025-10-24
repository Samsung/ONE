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

#ifndef __API_GGMA_PKG_H__
#define __API_GGMA_PKG_H__

#include "config.h"
#include <memory>
#include <string>

namespace ggma
{
class GGMATokenizer; // Forward declaration

class package
{
public:
  package(const char *path);
  ~package() = default;
  std::string path() const { return _path; }
  const ggma::GGMATokenizer *get_tokenizer() const;
  ggma::GGMAConfig load_config() const;

private:
  std::string _path;
  mutable std::unique_ptr<ggma::GGMATokenizer> _tokenizer;
};

} // namespace ggma

#endif // __API_GGMA_PKG_H__
