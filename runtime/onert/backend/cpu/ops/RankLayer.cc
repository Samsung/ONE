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

#include "RankLayer.h"

#include "OperationUtils.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::Rank &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Shape::Input::INPUT)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::RankLayer>();

  fn->configure(ifm_tensor, ofm_tensor);

  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::Rank &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

RankLayer::RankLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void RankLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void RankLayer::run()
{
  int32_t *output_data = getBuffer<int32_t>(_output);
  output_data[0] = _input->getShape().rank();
}

} // namespace onert::backend::cpu::ops
