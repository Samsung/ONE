/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ReshapeLayer.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  // optional 2nd input
  IPortableTensor *shape_tensor = nullptr;

  if (node.getInputs().size() == 2)
  {
    const auto shape_index{node.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    shape_tensor = _tensor_reg->getPortableTensor(shape_index);
  }

  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, shape_tensor, output_tensor);
  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::Reshape &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

ReshapeLayer::ReshapeLayer() : _input(nullptr), _shape(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ReshapeLayer::reshapeGeneric()
{
  // output buffer equals to input buffer means that copy is not needed
  if (_output->buffer() != _input->buffer())
  {
    size_t count = _input->total_size();
    memcpy(_output->buffer(), _input->buffer(), count);
  }
}

void ReshapeLayer::configure(const IPortableTensor *input, const IPortableTensor *shape,
                             IPortableTensor *output)
{
  _input = input;
  /* note : shape is optional. If not provided from model, _shape is nullptr. */
  _shape = shape;
  _output = output;
}

void ReshapeLayer::run() { reshapeGeneric(); }

} // namespace onert::backend::cpu::ops
