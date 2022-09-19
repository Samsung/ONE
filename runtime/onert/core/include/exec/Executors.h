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
#include "util/Index.h"

namespace onert
{
namespace exec
{

struct ExecutorIndexTag;
using ExecutorIndex = ::onert::util::Index<uint32_t, ExecutorIndexTag>;

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

  ExecutorIndex emplace(std::unique_ptr<IExecutor> exec, const ir::ModelIndex &model_index,
                        const ir::SubgraphIndex &subg_index);

  IExecutor *at(const ExecutorIndex &idx) { return _executors.at(idx).get(); }

  IExecutor *at(const ir::ModelIndex &idx_m, const ir::SubgraphIndex &idx_subg);

  uint32_t inputSize() const;

  uint32_t outputSize() const;

  const ir::OperandInfo inputInfo(const ir::IOIndex &index);

  const ir::OperandInfo outputInfo(const ir::IOIndex &index);

  void execute(const IODescription &desc);

private:
  void executeEntries(const IODescription &desc);

  ExecutorIndex generateIndex()
  {
    // No need to check if there is an entry with _next_index since
    // _next_index is always ("the highest index in the object map" + 1)
    if (ExecutorIndex{_next_index}.valid())
      return ExecutorIndex{_next_index++};
    else
      return ExecutorIndex{};
  }

private:
  std::unordered_map<ExecutorIndex, std::unique_ptr<IExecutor>> _executors;
  // NOTE _model_edges may use different struct type for executor implementation
  std::unique_ptr<ir::ModelEdges> _model_edges;
  std::unordered_map<ExecutorIndex, std::pair<ir::ModelIndex, ir::SubgraphIndex>> _index_map;
  uint32_t _next_index = 0;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
