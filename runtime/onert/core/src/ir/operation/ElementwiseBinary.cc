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

#include "ir/operation/ElementwiseBinary.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void ElementwiseBinary::accept(OperationVisitor &v) const { v.visit(*this); }

ElementwiseBinary::ElementwiseBinary(const OperandIndexSequence &inputs,
                                     const OperandIndexSequence &outputs, const Param &param)
  : Operation{OperandConstraint::createExact(2u), inputs, outputs}, _param{param}
{
}

std::string ElementwiseBinary::name() const
{
  using ElementwiseBinaryType = onert::ir::operation::ElementwiseBinary::ElementwiseBinaryType;
  static const std::unordered_map<ElementwiseBinaryType, std::string> name_map{
    {ElementwiseBinaryType::FLOOR_DIV, std::string{"FloorDiv"}},
    {ElementwiseBinaryType::FLOOR_MOD, std::string{"FloorMod"}},
    {ElementwiseBinaryType::LOGICAL_AND, std::string{"LogicalAnd"}},
    {ElementwiseBinaryType::LOGICAL_OR, std::string{"LogicalOr"}},
    {ElementwiseBinaryType::MAX, std::string{"Max"}},
    {ElementwiseBinaryType::MIN, std::string{"Min"}}};
  return name_map.at(_param.op_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
