/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/If.h"

#include <cstring>

namespace luci_interpreter
{
namespace kernels
{

If::If(const Tensor *cond, std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs,
       RuntimeGraph *then_graph, RuntimeGraph *else_graph)
    : _cond(cond), _inputs(std::move(inputs)), _outputs(std::move(outputs)),
      _then_graph(then_graph), _else_graph(else_graph)
{
}

void If::configure()
{
  assert(_cond->element_type() == DataType::BOOL);
  assert(_cond->shape().num_elements() == 1);

  for (RuntimeGraph *graph : {_then_graph, _else_graph})
  {
    assert(graph->getInputTensors().size() == _inputs.size());
    assert(graph->getOutputTensors().size() == _outputs.size());
  }
}

void If::execute() const
{
  const bool cond_value = _cond->data<bool>()[0];

  RuntimeGraph *active_graph = cond_value ? _then_graph : _else_graph;
  const auto &graph_inputs = active_graph->getInputTensors();
  const auto &graph_outputs = active_graph->getOutputTensors();

  // Copy kernel inputs to active graph inputs.
  for (size_t i = 0; i < _inputs.size(); ++i)
  {
    assert(graph_inputs[i]->element_type() == _inputs[i]->element_type());
    graph_inputs[i]->resize(_inputs[i]->shape());

    const int32_t num_elements = _inputs[i]->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(_inputs[i]->element_type());
    std::memcpy(graph_inputs[i]->data<void>(), _inputs[i]->data<void>(),
                num_elements * element_size);
  }

  active_graph->execute();

  // Copy graph outputs to kernel outputs.
  for (size_t i = 0; i < _outputs.size(); ++i)
  {
    assert(graph_outputs[i]->element_type() == _outputs[i]->element_type());
    _outputs[i]->resize(graph_outputs[i]->shape());

    const int32_t num_elements = _outputs[i]->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(_outputs[i]->element_type());
    std::memcpy(_outputs[i]->data<void>(), graph_outputs[i]->data<void>(),
                num_elements * element_size);
  }
}

} // namespace kernels
} // namespace luci_interpreter
