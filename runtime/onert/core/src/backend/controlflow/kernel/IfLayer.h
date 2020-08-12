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

#ifndef __ONERT_BACKEND_CONTROLFLOW_KERNEL_IF_LAYER_H__
#define __ONERT_BACKEND_CONTROLFLOW_KERNEL_IF_LAYER_H__

#include <backend/ITensor.h>
#include <exec/IExecutor.h>

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

class IfLayer : public ::onert::exec::IFunction
{
public:
  IfLayer(const std::shared_ptr<backend::ITensor> &cond_tensor,
          const std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
          const std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
          const ir::OperandIndexSequence &output_indices, const ir::Graph &graph,
          const exec::DynAllocInfoMap &outputs_dyn_alloc_info,
          const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
          exec::ExecutorMap *executor_map);

public:
  void run() override;

  const backend::ITensor *getOutput(int output_ind) const override
  {
    assert(static_cast<size_t>(output_ind) < _output_tensors.size());
    return _output_tensors[output_ind].get();
  }

private:
  const std::shared_ptr<backend::ITensor> _cond_tensor;
  const std::vector<std::shared_ptr<backend::ITensor>> _input_tensors;
  const std::vector<std::shared_ptr<backend::ITensor>> _output_tensors;
  const ir::OperandIndexSequence &_output_indices;
  const ir::Graph &_graph;
  const exec::DynAllocInfoMap _outputs_dyn_alloc_info;
  const ir::SubgraphIndex _then_subg_index;
  const ir::SubgraphIndex _else_subg_index;
  exec::ExecutorMap *_executor_map;
};

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_KERNEL_IF_LAYER_H__
