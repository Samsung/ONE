/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#include "kernels/Utils.h"

#include <cstring>

namespace luci_interpreter
{
namespace kernels
{

static std::vector<const Tensor *> joinInputs(const Tensor *cond,
                                              const std::vector<const Tensor *> &inputs)
{
  std::vector<const Tensor *> result{cond};
  result.insert(result.cend(), inputs.cbegin(), inputs.cend());
  return result;
}

If::If(const Tensor *cond, const std::vector<const Tensor *> &inputs, std::vector<Tensor *> outputs,
       RuntimeGraph *then_graph, RuntimeGraph *else_graph)
  : Kernel(joinInputs(cond, inputs), std::move(outputs)), _then_graph(then_graph),
    _else_graph(else_graph)
{
}

void If::configure()
{
  LUCI_INTERPRETER_CHECK(cond()->element_type() == DataType::BOOL);
  LUCI_INTERPRETER_CHECK(cond()->shape().num_elements() == 1);

  for (RuntimeGraph *graph : {_then_graph, _else_graph})
  {
    (void)graph;
    LUCI_INTERPRETER_CHECK(graph->getInputTensors().size() == getInputTensors().size() - 1);
    LUCI_INTERPRETER_CHECK(graph->getOutputTensors().size() == getOutputTensors().size());
  }
}

void If::execute() const
{
  const bool cond_value = cond()->data<bool>()[0];

  RuntimeGraph *active_graph = cond_value ? _then_graph : _else_graph;
  const auto &graph_inputs = active_graph->getInputTensors();
  const auto &graph_outputs = active_graph->getOutputTensors();

  // Copy kernel inputs to active graph inputs.
  for (size_t i = 0; i < getInputTensors().size() - 1; ++i)
  {
    LUCI_INTERPRETER_CHECK(graph_inputs[i]->element_type() == input(i)->element_type());
    graph_inputs[i]->resize(input(i)->shape());

    const int32_t num_elements = input(i)->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(input(i)->element_type());
    // TODO: Think about how allocate memory for output in main graph
    active_graph->configureAllocations(graph_inputs[i]);
    std::memcpy(graph_inputs[i]->data<void>(), input(i)->data<void>(), num_elements * element_size);
  }

  active_graph->execute();

  // Copy graph outputs to kernel outputs.
  for (size_t i = 0; i < getOutputTensors().size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(graph_outputs[i]->element_type() == output(i)->element_type());
    output(i)->resize(graph_outputs[i]->shape());
    // TODO: Think about how allocate memory for output in main graph
    active_graph->configureAllocations(output(i));

    const int32_t num_elements = output(i)->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(output(i)->element_type());
    std::memcpy(output(i)->data<void>(), graph_outputs[i]->data<void>(),
                num_elements * element_size);
  }
}

} // namespace kernels
} // namespace luci_interpreter
