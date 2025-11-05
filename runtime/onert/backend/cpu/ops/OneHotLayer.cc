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

#include "OneHotLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/OneHot.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::OneHot &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::OneHot &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto indices_index{node.getInputs().at(ir::operation::OneHot::INDICES)};
  const auto depth_index{node.getInputs().at(ir::operation::OneHot::Input::DEPTH)};
  const auto onvalue_index{node.getInputs().at(ir::operation::OneHot::Input::ON_VALUE)};
  const auto offvalue_index{node.getInputs().at(ir::operation::OneHot::Input::OFF_VALUE)};

  const auto axis = node.param().axis;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto indices_tensor = _tensor_reg->getPortableTensor(indices_index);
  auto depth_tensor = _tensor_reg->getPortableTensor(depth_index);
  auto onvalue_tensor = _tensor_reg->getPortableTensor(onvalue_index);
  auto offvalue_tensor = _tensor_reg->getPortableTensor(offvalue_index);

  assert(indices_tensor->data_type() == OperandType::INT32);
  assert(axis <= static_cast<int>(indices_tensor->getShape().rank()));

  auto fn = std::make_unique<ops::OneHotLayer>();

  fn->configure(indices_tensor, depth_tensor, onvalue_tensor, offvalue_tensor, output_tensor, axis);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

template <typename T> void OneHotLayer::oneHotImpl()
{
  // It assumes index is int32_t type.
  nnfw::cker::OneHot<T, int32_t>(
    *getBuffer<int32_t>(_depth), *getBuffer<T>(_on_value), *getBuffer<T>(_off_value), _axis,
    getShape(_indices), getBuffer<int32_t>(_indices), getShape(_output), getBuffer<T>(_output));
}

void OneHotLayer::configure(const IPortableTensor *indices, const IPortableTensor *depth,
                            const IPortableTensor *on_value, const IPortableTensor *off_value,
                            IPortableTensor *output, const int32_t axis)
{
  _indices = indices;
  _output = output;
  _depth = depth;
  _on_value = on_value;
  _off_value = off_value;
  _axis = axis;
}

void OneHotLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
  {
    oneHotImpl<float>();
  }
  else
  {
    throw std::runtime_error{"OneHot: unsupported data type"};
  }
}

} // namespace onert::backend::cpu::ops
