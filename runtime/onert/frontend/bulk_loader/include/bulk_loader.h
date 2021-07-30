/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __BULK_LOADER_H_
#define __BULK_LOADER_H_

#include "ir/Graph.h"
#include "ir/Shape.h"
#include "ir/Operations.Include.h"

namespace onert
{
namespace bulk_loader
{

class BulkLoader
{
public:
  BulkLoader(std::unique_ptr<ir::Subgraphs> &subgraphs) : _subgraphs{subgraphs} {}
  void loadFromFile(const std::string &file_path);

private:
  std::unique_ptr<ir::Subgraphs> &_subgraphs;
};

void BulkLoader::loadFromFile(const std::string &file_path)
{
  auto subgraphs = std::make_unique<ir::Subgraphs>();

  ir::operation::Bulk::Param param;
  param.path = file_path;

  // TODO: need I/O information for bulk
}

std::unique_ptr<ir::Subgraphs> loadModel(const std::string &filename)
{
  auto subgraphs = std::make_unique<ir::Subgraphs>();
  BulkLoader loader(subgraphs);
  loader.loadFromFile(filename);
  return subgraphs;
}

} // namespace bulk_loader
} // namespace onert
#endif // __BULK_LOADER_H_
