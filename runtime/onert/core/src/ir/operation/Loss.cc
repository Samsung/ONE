/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/operation/Loss.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void Loss::accept(OperationVisitor &v) const { v.visit(*this); }

Loss::Loss(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
           const Param &param)
  : Operation{OperandConstraint::createAtLeast(2u), inputs, outputs}, _param{param}
{
  if (param.op_type == Type::CATEGORICAL_CROSSENTROPY)
  {
    assert(inputs.size() == 2 && "CategoricalCrossentropy Loss has 2 inputs");
  }
}

std::string Loss::name() const
{
  using LossType = onert::ir::operation::Loss::Type;
  static const std::unordered_map<Type, std::string> name_map{
    {LossType::CATEGORICAL_CROSSENTROPY, "CategoricalCrossentropy Loss"}};
  return name_map.at(_param.op_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
