/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "trix_loader.h"

#include <libnpuhost.h>
#include <npubinfmt.h>
#include <typedef.h>

namespace onert
{
namespace trix_loader
{

bool TrixLoader::loadModel()
{
  auto meta = getNPUmodel_metadata(_model_path.c_str(), false);
  if (meta == nullptr)
  {
    std::cerr << "ERROR: Failed to get TRIV2 model metadata" << std::endl;
    return false;
  }

  if (NPUBIN_VERSION(meta->magiccode) != 3)
  {
    return false;
  }
  return true;
}

std::unique_ptr<ir::Subgraphs> loadModel(const std::string &filename)
{
  auto subgraphs = std::make_unique<ir::Subgraphs>();
  TrixLoader loader(subgraphs);
  loader.loadFromFile(filename);
  return subgraphs;
}

} // namespace trix_loader
} // namespace onert
