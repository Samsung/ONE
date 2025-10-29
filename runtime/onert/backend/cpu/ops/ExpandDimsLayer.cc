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

#include "ExpandDimsLayer.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::ExpandDims &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::ExpandDims &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ExpandDims::Input::INPUT)};
  // AXIS input is used for output shape inference

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto fn = std::make_unique<ops::ExpandDimsLayer>();

  fn->configure(input_tensor, output_tensor);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

ExpandDimsLayer::ExpandDimsLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ExpandDimsLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void ExpandDimsLayer::run()
{
  // output buffer equals to input buffer means that copy is not needed
  if (_output->buffer() != _input->buffer())
  {
    size_t count = _input->total_size();
    memcpy(_output->buffer(), _input->buffer(), count);
  }
}

} // namespace onert::backend::cpu::ops
