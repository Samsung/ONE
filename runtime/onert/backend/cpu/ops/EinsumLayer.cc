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

#include "EinsumLayer.h"

#include <cker/operation/Einsum.h>

namespace onert::backend::cpu::ops
{

EinsumLayer::EinsumLayer()
  : _inputs(), _output(nullptr), _equation(), _einsum_kernel(new nnfw::cker::Einsum())
{
  // DO NOTHING
}

EinsumLayer::~EinsumLayer() = default;

void EinsumLayer::einsumFloat32()
{
  uint32_t num_inputs = _inputs.size();
  nnfw::cker::Einsum &kernel = *_einsum_kernel;

  kernel.prepare(_equation);

  std::vector<nnfw::cker::Shape> inputShapes;
  std::vector<const float *> inputFloatPtrs;

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    inputShapes.emplace_back(getShape(_inputs[i]));
    inputFloatPtrs.emplace_back(getBuffer<float>(_inputs[i]));
  }

  kernel(_equation, inputShapes, inputFloatPtrs, getShape(_output), getBuffer<float>(_output));
}

void EinsumLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    einsumFloat32();
  }
  else
  {
    throw std::runtime_error{"Einsum: unsupported data type"};
  }
}

void EinsumLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                            std::string equation, IPortableTensor *output)
{
  assert(inputs.size() > 0);
  assert(output != nullptr);

  _inputs = inputs;
  _equation = equation;
  _output = output;
}

} // namespace onert::backend::cpu::ops
