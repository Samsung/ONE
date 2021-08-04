/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/While.h"
#include "kernels/Utils.h"

#include <cstring>

namespace luci_interpreter
{
namespace kernels
{

While::While(std::vector<const Tensor *> inputs, std::vector<Tensor *> outputs,
             RuntimeGraph *cond_graph, RuntimeGraph *body_graph)
  : Kernel(std::move(inputs), std::move(outputs)), _cond_graph(cond_graph), _body_graph(body_graph)
{
}

void While::configure()
{
  LUCI_INTERPRETER_CHECK(_body_graph->getInputTensors().size() == getInputTensors().size());
  LUCI_INTERPRETER_CHECK(_body_graph->getOutputTensors().size() == getOutputTensors().size());
  LUCI_INTERPRETER_CHECK(_body_graph->getOutputTensors().size() == getInputTensors().size());

  LUCI_INTERPRETER_CHECK(_cond_graph->getInputTensors().size() == getInputTensors().size());

  const auto &cond_outputs = _cond_graph->getOutputTensors();
  LUCI_INTERPRETER_CHECK(cond_outputs.size() == 1)
  LUCI_INTERPRETER_CHECK(cond_outputs[0]->element_type() == DataType::BOOL);
}

/**
 * @note Dynamic shape such as {1, 0, 8} may fail in tensor->data()
 */
void While::execute() const
{
  const auto &cond_inputs = _cond_graph->getInputTensors();
  const auto &cond_outputs = _cond_graph->getOutputTensors();

  // Copy initial kernel inputs to condition graph inputs.
  for (size_t i = 0; i < getInputTensors().size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(cond_inputs[i]->element_type() == input(i)->element_type());
    cond_inputs[i]->resize(input(i)->shape());

    const int32_t num_elements = input(i)->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(input(i)->element_type());
    std::memcpy(cond_inputs[i]->data<void>(), input(i)->data<void>(), num_elements * element_size);
  }

  const auto &body_inputs = _body_graph->getInputTensors();
  const auto &body_outputs = _body_graph->getOutputTensors();

  while (true)
  {
    _cond_graph->execute();

    bool cond_value = cond_outputs[0]->data<bool>()[0];
    if (!cond_value)
      break;

    // Copy cond subgraph inputs to body subgraph inputs.
    for (size_t i = 0; i < _cond_graph->getInputTensors().size(); ++i)
    {
      LUCI_INTERPRETER_CHECK(body_inputs[i]->element_type() == cond_inputs[i]->element_type());
      body_inputs[i]->resize(cond_inputs[i]->shape());

      const int32_t num_elements = cond_inputs[i]->shape().num_elements();
      const std::size_t element_size = getDataTypeSize(cond_inputs[i]->element_type());
      std::memcpy(body_inputs[i]->data<void>(), cond_inputs[i]->data<void>(),
                  num_elements * element_size);
    }

    _body_graph->execute();

    // Copy body subgraph outputs to cond subgraph inputs
    for (size_t i = 0; i < _cond_graph->getInputTensors().size(); ++i)
    {
      LUCI_INTERPRETER_CHECK(body_outputs[i]->element_type() == cond_inputs[i]->element_type());
      cond_inputs[i]->resize(body_outputs[i]->shape());

      const int32_t num_elements = body_outputs[i]->shape().num_elements();
      const std::size_t element_size = getDataTypeSize(body_outputs[i]->element_type());
      std::memcpy(cond_inputs[i]->data<void>(), body_outputs[i]->data<void>(),
                  num_elements * element_size);
    }
  }

  // Copy cond subgraph outputs to this kernel outputs.
  for (size_t i = 0; i < getOutputTensors().size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(cond_inputs[i]->element_type() == output(i)->element_type());
    output(i)->resize(cond_inputs[i]->shape());

    const int32_t num_elements = output(i)->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(output(i)->element_type());
    std::memcpy(output(i)->data<void>(), cond_inputs[i]->data<void>(), num_elements * element_size);
  }
}

} // namespace kernels
} // namespace luci_interpreter
