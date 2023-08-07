/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_TRAIN_LOWERED_TRAINABLE_GRAPH_H__
#define __ONERT_COMPILER_TRAIN_LOWERED_TRAINABLE_GRAPH_H__

#include "compiler/BackendResolver.h"
#include "compiler/CompilerOptions.h"
#include "compiler/GraphLowerInfo.h"
#include "compiler/ILoweredGraph.h"
#include "ir/train/TrainableGraph.h"

namespace onert
{
namespace compiler
{
namespace train
{

// TODO Unify with LoweredGraph
/**
 * @brief Class that contains lowering information on graph.
 *        In addition, after lowering, operands in graph will be set to "dynamic"
 *        if the shape of output of an operation cannot be decided at compilation time.
 */
class LoweredTrainableGraph : public ILoweredGraph
{
public:
  LoweredTrainableGraph(ir::train::TrainableGraph &graph, const compiler::CompilerOptions &options);

  // TODO Remove const_cast
  ir::Graph &graph() override { return const_cast<ir::Graph &>(_trainable_graph.graph()); }
  const ir::Graph &graph() const override { return _trainable_graph.graph(); }
  ir::train::TrainableGraph &trainable_graph() { return _trainable_graph; }
  const ir::train::TrainableGraph &trainable_graph() const { return _trainable_graph; }
  const compiler::GraphLowerInfo &lower_info() const override { return _lower_info_map; }
  compiler::GraphLowerInfo &lower_info() override { return _lower_info_map; }
  std::shared_ptr<ir::OperationIndexMap<int64_t>> indexed_ranks() { return _indexed_ranks; }

  void setHasDynamicTensor(ir::OperationIndex, bool has_dynamic) override
  {
    if (has_dynamic)
      throw std::runtime_error("LoweredTrainableGraph does not support dynamic tensors yet");
  }
  bool getHasDynamicTensor(ir::OperationIndex) const override { return false; }

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
  ir::train::TrainableGraph _trainable_graph;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  compiler::GraphLowerInfo _lower_info_map;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_LOWERED_TRAINABLE_GRAPH_H__
