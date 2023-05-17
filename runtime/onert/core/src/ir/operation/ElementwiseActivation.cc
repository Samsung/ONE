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

#include "ir/operation/ElementwiseActivation.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void ElementwiseActivation::accept(OperationVisitor &v) const { v.visit(*this); }
void ElementwiseActivation::accept(MutableOperationVisitor &v) { v.visit(*this); }

ElementwiseActivation::ElementwiseActivation(const OperandIndexSequence &inputs,
                                             const OperandIndexSequence &outputs,
                                             const Param &param)
  : Operation{OperandConstraint::createExact(1u), inputs, outputs}, _param{param}
{
  if (param.op_type == Type::LOGISTIC)
  {
    assert(param.alpha == 0.0f && param.beta == 0.0f &&
           "Logistic will be supported only as "
           "sigmoid function(L=1, k=1, x0=0). So, do "
           "not use alpha and beta");
  }
  else if (param.op_type == Type::RELU)
  {
    assert(param.alpha >= param.beta && "ReLU's alpha must be equal or greater than beta");
  }
  else if (param.op_type == Type::TANH)
  {
    assert(param.alpha == 1.0f && param.beta == 1.0f &&
           "f(x) = alpha * tanh(beta * x), Tanh is "
           "supported only the values of alpha and "
           "beta are 1.f");
  }
}

std::string ElementwiseActivation::name() const
{
  using ElementwiseActivationType = onert::ir::operation::ElementwiseActivation::Type;
  static const std::unordered_map<Type, std::string> name_map{
    {ElementwiseActivationType::ELU, "ELU"},
    {ElementwiseActivationType::LOGISTIC, "Logistic"},
    {ElementwiseActivationType::RELU, "ReLU"},
    {ElementwiseActivationType::TANH, "Tanh"},
    {ElementwiseActivationType::LEAKY_RELU, "LeakyRelu"}};
  return name_map.at(_param.op_type);
}

float ElementwiseActivation::infinity = std::numeric_limits<float>::infinity();

} // namespace operation
} // namespace ir
} // namespace onert
