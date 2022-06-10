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

#ifndef __ONERT_IR_MODEL_GRAPH_H__
#define __ONERT_IR_MODEL_GRAPH_H__

#include <memory>

#include "ir/Index.h"
#include "ir/Models.h"
#include "ir/Subgraphs.h"

namespace onert
{
namespace ir
{

class ModelGraph
{
public:
  ModelGraph() = default;
  ModelGraph(const ModelGraph &obj) = default;
  ModelGraph(ModelGraph &&) = default;
  ModelGraph &operator=(const ModelGraph &) = default;
  ModelGraph &operator=(ModelGraph &&) = default;
  ~ModelGraph() = default;

  const Models &models() const { return _models; }
  Models &models() { return _models; }
  std::shared_ptr<Subgraphs> &entry() { return _models.at(onert::ir::ModelIndex{0})._subgraphs; }

private:
  Models _models;
  // TODO: Add connection between models
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_MODEL_GRAPH_H__
