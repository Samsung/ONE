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

#include "package.h"
#include "tokenize.h"

#include <filesystem>
#include <string>

namespace ggma
{

package::package(const char *path) : _path(path) {}

const ggma::GGMATokenizer *package::get_tokenizer() const
{
  if (!_tokenizer)
    _tokenizer = std::make_unique<ggma::GGMATokenizer>();
  return _tokenizer.get();
}

ggma::GGMAConfig package::load_config() const
{
  ggma::GGMAConfig config;
  std::filesystem::path pkg_path(_path);
  std::filesystem::path model_config_file = pkg_path / "config.json";
  std::filesystem::path ggma_config_file = pkg_path / "ggma.config.json";

  // TODO: Load ggma_config from ggma_config_file if necessary
  // For now, we only load model_config as per the original logic.
  config.model.load_from_file(model_config_file);
  return config;
}

} // namespace ggma
