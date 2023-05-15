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

#include "ir/operation/BinaryArithmetic.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void BinaryArithmetic::accept(OperationVisitor &v) const { v.visit(*this); }
void BinaryArithmetic::accept(MutableOperationVisitor &v) { v.visit(*this); }

BinaryArithmetic::BinaryArithmetic(const OperandIndexSequence &inputs,
                                   const OperandIndexSequence &outputs, const Param &param)
  : Operation{OperandConstraint::createExact(2u), inputs, outputs}, _param{param}
{
}

std::string BinaryArithmetic::name() const
{
  using ArithmeticType = onert::ir::operation::BinaryArithmetic::ArithmeticType;
  static const std::unordered_map<ArithmeticType, std::string> name_map{
    {ArithmeticType::ADD, std::string{"Add"}},
    {ArithmeticType::SUB, std::string{"Sub"}},
    {ArithmeticType::MUL, std::string{"Mul"}},
    {ArithmeticType::DIV, std::string{"Div"}}};
  return name_map.at(_param.arithmetic_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
