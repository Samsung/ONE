/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BinaryArithmeticLayer.h"

#include "OperationUtils.h"

#include <cker/Shape.h>
#include <cker/train/operation/BinaryArithmetic.h>
#include <cker/operation/BinaryArithmeticOps.h>
#include <cker/train/operation/BinaryArithmetic.h>
#include <cker/train/operation/ReLU.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

BinaryArithmeticLayer::BinaryArithmeticLayer()
  : cpu::ops::BinaryArithmeticLayer(), _back_prop_lhs{nullptr}, _back_prop_rhs{nullptr},
    _back_prop_output{nullptr}
{
  // DO NOTHING
}

void BinaryArithmeticLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                                      IPortableTensor *output, IPortableTensor *back_prop_lhs,
                                      IPortableTensor *back_prop_rhs,
                                      const IPortableTensor *back_prop_output,
                                      const ir::Activation activation,
                                      const ArithmeticType arithmetic_type)
{
  cpu::ops::BinaryArithmeticLayer::configure(
    lhs, rhs, output, activation, static_cast<cpu::ops::ArithmeticType>(arithmetic_type));

  _back_prop_lhs = back_prop_lhs;
  _back_prop_rhs = back_prop_rhs;
  _back_prop_output = back_prop_output;
  _arithmetic_type = arithmetic_type;
  _activation = activation;
}

void BinaryArithmeticLayer::forward(bool) { cpu::ops::BinaryArithmeticLayer::run(); }

void BinaryArithmeticLayer::backward()
{
  // Calculate gradient for activation
  if (_back_prop_output->data_type() != OperandType::FLOAT32)
    throw std::runtime_error{"Unsupported Data Type"};

  const IPortableTensor *backprop_act;
  try
  {
    backprop_act =
      backpropActivation(_activation, _output, _back_prop_output, _act_back_prop_output.get());
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error{"train BinaryArithmeticLayer: " + std::string(e.what())};
  }
  assert(backprop_act != nullptr);

  nnfw::cker::train::BinaryArithmeticGrad(
    getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_back_prop_lhs),
    getBuffer<float>(_back_prop_lhs), getShape(_back_prop_rhs), getBuffer<float>(_back_prop_rhs),
    static_cast<nnfw::cker::train::ArithmeticType>(_arithmetic_type));
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
