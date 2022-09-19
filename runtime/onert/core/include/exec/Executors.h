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

  void emplace(const ir::SubgraphIndex &idx, std::unique_ptr<IExecutor> exec);

  IExecutor *at(const ExecutorIndex &idx) const { return _executors.at(idx).get(); }

  IExecutor *at(const ir::SubgraphIndex &idx) const;

  uint32_t inputSize() const;

  uint32_t outputSize() const;

  const ir::OperandInfo inputInfo(const ir::IOIndex &index);

  const ir::OperandInfo outputInfo(const ir::IOIndex &index);

  void execute(const IODescription &desc);

private:
  void executeEntries(const IODescription &desc);

  /**
   * @brief   Generate a executor index with `_next_index`
   * @return  Generated executor index
   * @note    This is copy from util/ObjectManager.h
   */
  ExecutorIndex generateIndex()
  {
    if (ExecutorIndex{_next_index}.valid())
      return ExecutorIndex{_next_index++};
    else
      return ExecutorIndex{};
  }

private:
  // NOTE _executors[0] is primary executor
  std::unordered_map<ExecutorIndex, std::unique_ptr<IExecutor>> _executors;
  // NOTE _model_edges may use different struct type for executor implementation
  std::unique_ptr<ir::ModelEdges> _model_edges;
  // TODO Use ModelIndex and SubgraphIndex pair
  std::unordered_map<ExecutorIndex, ir::SubgraphIndex> _index_map;
  uint32_t _next_index = 0;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
