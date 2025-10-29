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

#include "ShapeLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Shape &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Shape &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Shape::Input::INPUT)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::ShapeLayer>();

  fn->configure(ifm_tensor, ofm_tensor);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

ShapeLayer::ShapeLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

template <typename T> void GetRawShape(const IPortableTensor *input, T *output_data)
{
  auto shape = input->getShape();
  for (int i = 0; i < shape.rank(); ++i)
  {
    output_data[i] = static_cast<T>(shape.dim(i));
  }
}

void ShapeLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void ShapeLayer::run()
{
  if (_output->data_type() == OperandType::UINT32)
  {
    GetRawShape(_input, getBuffer<uint32_t>(_output));
  }
  else if (_output->data_type() == OperandType::INT32)
  {
    GetRawShape(_input, getBuffer<int32_t>(_output));
  }
  else if (_output->data_type() == OperandType::INT64)
  {
    GetRawShape(_input, getBuffer<int64_t>(_output));
  }
  else
  {
    throw std::runtime_error{"NYI : not supported output type for ShapeLayer"};
  }
}

} // namespace onert::backend::cpu::ops
