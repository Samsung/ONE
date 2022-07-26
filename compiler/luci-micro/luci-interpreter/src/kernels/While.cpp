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

namespace
{

void copy(const std::vector<const Tensor *> &src, const std::vector<Tensor *> &dst)
{
  for (size_t i = 0; i < src.size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(dst[i]->element_type() == src[i]->element_type());
    dst[i]->resize(src[i]->shape());

    const int32_t num_elements = src[i]->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(src[i]->element_type());
    std::memcpy(dst[i]->data<void>(), src[i]->data<void>(), num_elements * element_size);
  }
}

void copy(const std::vector<Tensor *> &src, const std::vector<Tensor *> &dst)
{
  std::vector<const Tensor *> const_src;
  for (const auto &t : src)
    const_src.push_back(t);
  copy(const_src, dst);
}

// TODO: Think about how allocate memory for output in main graph
void configureTensorsAllocations(const std::vector<Tensor *> &tensors, RuntimeGraph *run_graph)
{
  for (auto tensor : tensors)
    run_graph->configureAllocations(tensor);
}

} // namespace

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

  configureTensorsAllocations(cond_inputs, _cond_graph);

  copy(getInputTensors(), cond_inputs);

  const auto &body_inputs = _body_graph->getInputTensors();
  const auto &body_outputs = _body_graph->getOutputTensors();

  configureTensorsAllocations(body_inputs, _body_graph);

  while (true)
  {
    _cond_graph->execute();

    bool cond_value = cond_outputs[0]->data<bool>()[0];
    if (!cond_value)
      break;

    copy(cond_inputs, body_inputs);

    _body_graph->execute();

    copy(body_outputs, cond_inputs);
  }

  copy(cond_inputs, getOutputTensors());
}

} // namespace kernels
} // namespace luci_interpreter
