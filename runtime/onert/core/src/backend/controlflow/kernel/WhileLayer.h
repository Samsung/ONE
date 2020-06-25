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

#ifndef __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__
#define __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__

#include <backend/ITensor.h>
#include <exec/IExecutor.h>
#include <exec/IFunction.h>
#include <ir/OperandIndexSequence.h>
#include <ir/Graph.h>

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

class WhileLayer : public ::onert::exec::IFunction
{
public:
  WhileLayer(const std::vector<std::shared_ptr<backend::ITensor>> &input_tensors,
             const std::vector<std::shared_ptr<backend::ITensor>> &output_tensors,
             const ir::OperandIndexSequence &output_indices, const ir::Graph &graph,
             const exec::DynAllocInfoMap &outputs_dyn_alloc_info,
             const ir::SubgraphIndex &cond_subg_index, const ir::SubgraphIndex &body_subg_index,
             exec::ExecutorMap *executor_map);

public:
  void run() override;

private:
  const ir::SubgraphIndex _cond_subg_index;
  const ir::SubgraphIndex _body_subg_index;
  const ir::OperandIndexSequence &_output_indices;
  const ir::Graph &_graph;
  const std::vector<std::shared_ptr<backend::ITensor>> _input_tensors;
  const std::vector<std::shared_ptr<backend::ITensor>> _output_tensors;
  const exec::DynAllocInfoMap _outputs_dyn_alloc_info;
  exec::ExecutorMap *_executor_map;
};

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__
