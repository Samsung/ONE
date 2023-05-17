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

#include "ir/operation/Pool2D.h"
#include "ir/OperationVisitor.h"

#include <unordered_map>

namespace onert
{
namespace ir
{
namespace operation
{

void Pool2D::accept(OperationVisitor &v) const { v.visit(*this); }
void Pool2D::accept(MutableOperationVisitor &v) { v.visit(*this); }

Pool2D::Pool2D(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
               const Param &param)
  : Operation{OperandConstraint::createExact(1u), inputs, outputs}, _param{param}
{
}

std::string Pool2D::name() const
{
  using PoolType = onert::ir::operation::Pool2D::PoolType;
  static const std::unordered_map<PoolType, std::string> name_map{
    {PoolType::AVG, "Avg" + std::string{toString(opcode())}},
    {PoolType::L2, "L2" + std::string{toString(opcode())}},
    {PoolType::MAX, "Max" + std::string{toString(opcode())}}};
  return name_map.at(_param.op_type);
}

} // namespace operation
} // namespace ir
} // namespace onert
