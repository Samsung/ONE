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

#include "ir/operation/Reduce.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void Reduce::accept(OperationVisitor &v) const { v.visit(*this); }
void Reduce::accept(MutableOperationVisitor &v) { v.visit(*this); }

Reduce::Reduce(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
               const Param &param)
  : Operation{OperandConstraint::createExact(2u), inputs, outputs}, _param{param}
{
}

std::string Reduce::name() const
{
  using ReduceType = onert::ir::operation::Reduce::ReduceType;
  static const std::unordered_map<ReduceType, std::string> name_map{
    {ReduceType::ALL, std::string{toString(opcode())} + "All"},
    {ReduceType::ANY, std::string{toString(opcode())} + "Any"},
    {ReduceType::MAX, std::string{toString(opcode())} + "Max"},
    {ReduceType::MEAN, std::string{toString(opcode())} + "Mean"},
    {ReduceType::MIN, std::string{toString(opcode())} + "Min"},
    {ReduceType::PROD, std::string{toString(opcode())} + "Prod"},
    {ReduceType::SUM, std::string{toString(opcode())} + "SUM"}};
  return name_map.at(_param.reduce_type);
  //  return std::string(toString(opcode())) + reduce_type_str_map.at(_param.reduce_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
