/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_LOWERED_GRAPH_H__
#define __ONERT_COMPILER_LOWERED_GRAPH_H__

#include "compiler/BackendResolver.h"
#include "compiler/Compiler.h"
#include "compiler/GraphLowerInfo.h"
#include "compiler/ILoweredGraph.h"
#include "ir/Graph.h"

namespace onert
{
namespace compiler
{

/**
 * @brief Class that contains lowering information on graph.
 *        In addition, after lowering, operands in graph will be set to "dynamic"
 *        if the shape of output of an operation cannot be decided at compilation time.
 */
class LoweredGraph : public ILoweredGraph
{
public:
  LoweredGraph(const ir::Graph &graph, const compiler::CompilerOptions &options);

  ir::Graph &graph() override { return _graph; }
  const ir::Graph &graph() const override { return _graph; }
  const compiler::GraphLowerInfo &lower_info() const override { return _lower_info_map; }
  compiler::GraphLowerInfo &lower_info() override { return _lower_info_map; }

  std::shared_ptr<ir::OperationIndexMap<int64_t>> indexed_ranks() { return _indexed_ranks; }

  void setDynamicTensor(ir::OperationIndex ind, bool val) override
  {
    _has_dynamic_tensor_map.emplace(ind, val);
  }
  bool isDynamicTensor(ir::OperationIndex ind) const override
  {
    auto itr = _has_dynamic_tensor_map.find(ind);
    return (itr == _has_dynamic_tensor_map.end()) ? false : itr->second;
  }

private:
  void makeLowerInfo(const compiler::BackendResolver &backend_resolver);
  void dumpLowerInfo();
  void lowerGraph(const compiler::CompilerOptions &options);

private:
  /**
   *  @brief  Copy of target graph for lowering
   *  @note   It uses copy of graph, not reference.
   *          It allows the original graph can be compiled multiple times.
   */
  ir::Graph _graph;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  compiler::GraphLowerInfo _lower_info_map;
  ir::OperationIndexMap<bool> _has_dynamic_tensor_map;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_LOWERED_GRAPH_H__
