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

#include "ir/operation/ElementwiseUnary.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void ElementwiseUnary::accept(OperationVisitor &v) const { v.visit(*this); }
void ElementwiseUnary::accept(MutableOperationVisitor &v) { v.visit(*this); }

ElementwiseUnary::ElementwiseUnary(const OperandIndexSequence &inputs,
                                   const OperandIndexSequence &outputs, const Param &param)
  : Operation{OperandConstraint::createExact(1u), inputs, outputs,
              OperandConstraint::createExact(1u)},
    _param{param}
{
}

std::string ElementwiseUnary::name() const
{
  using ElementwiseUnaryType = onert::ir::operation::ElementwiseUnary::Type;
  static const std::unordered_map<ElementwiseUnaryType, std::string> name_map{
    {ElementwiseUnaryType::ABS, std::string{"Abs"}},
    {ElementwiseUnaryType::CAST, std::string{"Cast"}},
    {ElementwiseUnaryType::COS, std::string{"Cos"}},
    {ElementwiseUnaryType::DEQUANTIZE, std::string{"Dequantize"}},
    {ElementwiseUnaryType::ERF, std::string{"Erf"}},
    {ElementwiseUnaryType::EXP, std::string{"Exp"}},
    {ElementwiseUnaryType::FLOOR, std::string{"Floor"}},
    {ElementwiseUnaryType::LOG, std::string{"Log"}},
    {ElementwiseUnaryType::LOGICAL_NOT, std::string{"LogicalNot"}},
    {ElementwiseUnaryType::NEG, std::string{"Neg"}},
    {ElementwiseUnaryType::QUANTIZE, std::string{"Quantize"}},
    {ElementwiseUnaryType::ROUND, std::string{"Round"}},
    {ElementwiseUnaryType::RSQRT, std::string{"RSqrt"}},
    {ElementwiseUnaryType::SIN, std::string{"Sin"}},
    {ElementwiseUnaryType::SQRT, std::string{"Sqrt"}},
    {ElementwiseUnaryType::SQUARE, std::string{"Square"}},
    {ElementwiseUnaryType::ZEROS_LIKE, std::string{"ZerosLike"}}};
  return name_map.at(_param.op_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
