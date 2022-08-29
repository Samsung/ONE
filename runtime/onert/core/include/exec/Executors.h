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

#ifndef __ONERT_EXEC_EXECUTORS_H__
#define __ONERT_EXEC_EXECUTORS_H__

#include "IExecutor.h"
#include "ir/NNPkg.h"

namespace onert
{
namespace exec
{

/**
 * @brief Class to gather executors
 */
class Executors
{
public:
  Executors(void) = default;
  Executors(std::unique_ptr<ir::ModelEdges> model_edges) { _model_edges = std::move(model_edges); }
  Executors(const Executors &) = delete;
  Executors(Executors &&) = default;

  // TODO Use Executor index
  void emplace(ir::SubgraphIndex idx, std::unique_ptr<IExecutor> exec)
  {
    _executors.emplace(idx, std::move(exec));
  }

  std::unique_ptr<IExecutor> &at(ir::SubgraphIndex idx) { return _executors.at(idx); }

  uint32_t inputSize() const;

  uint32_t outputSize() const;

  const ir::OperandInfo inputInfo(const ir::IOIndex &index);

  const ir::OperandInfo outputInfo(const ir::IOIndex &index);

  void execute(const IODescription &desc);

private:
  // TODO Use Executor index
  //      Changing index will effect if/while compile and kernel implementation
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<IExecutor>> _executors;
  // NOTE _model_edges may use different struct type for executor implementation
  std::unique_ptr<ir::ModelEdges> _model_edges;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
