/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/operation/TopKV2.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace ir
{
namespace operation
{

void TopKV2::accept(OperationVisitor &v) const { v.visit(*this); }
void TopKV2::accept(MutableOperationVisitor &v) { v.visit(*this); }

TopKV2::TopKV2(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
               const Param &param)
  : Operation{OperandConstraint::createExact(1u), inputs, outputs}, _param{param}
{
}

} // namespace operation
} // namespace ir
} // namespace onert
