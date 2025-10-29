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

#include "StatelessRandomUniformLayer.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/StatelessRandomUniform.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::StatelessRandomUniform &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::StatelessRandomUniform &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto shape_index{node.getInputs().at(ir::operation::StatelessRandomUniform::SHAPE)};
  const auto seed_index{node.getInputs().at(ir::operation::StatelessRandomUniform::SEED)};

  auto output_alloc = _tensor_reg->getPortableTensor(output_index);
  auto shape_alloc = _tensor_reg->getPortableTensor(shape_index);
  auto seed_alloc = _tensor_reg->getPortableTensor(seed_index);

  auto fn = std::make_unique<ops::StatelessRandomUniformLayer>();

  fn->configure(shape_alloc, seed_alloc, output_alloc);
  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

StatelessRandomUniformLayer::StatelessRandomUniformLayer()
  : _shape(nullptr), _seed(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void StatelessRandomUniformLayer::configure(const IPortableTensor *shape,
                                            const IPortableTensor *seed, IPortableTensor *output)
{
  _shape = shape;
  _seed = seed;
  _output = output;
}

void StatelessRandomUniformLayer::StatelessRandomUniformFloat32()
{
  nnfw::cker::StatelessRandomUniform(getShape(_shape), getBuffer<int32_t>(_shape), getShape(_seed),
                                     getBuffer<int32_t>(_seed), getShape(_output),
                                     getBuffer<float>(_output));
}

void StatelessRandomUniformLayer::run()
{
  switch (_output->data_type())
  {
    // ToDo : It need to support INT8 and UINT8 also when will be applied quantization.
    case OperandType::FLOAT32:
      StatelessRandomUniformFloat32();
      break;
    default:
      throw std::runtime_error{"StatelessRandomUniformLayer: unsupported data type"};
  }
}

} // namespace onert::backend::cpu::ops
