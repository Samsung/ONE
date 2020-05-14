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

#include "KernelGenerator.h"

#include <util/Utils.h>

namespace onert
{
namespace backend
{
namespace controlflow
{

KernelGenerator::KernelGenerator(const ir::Operands &operand_ctx)
    : _operand_ctx{operand_ctx}, _tensor_builder_set{nullptr}, _executor_map{nullptr}
{
  UNUSED_RELEASE(_operand_ctx);
  UNUSED_RELEASE(_tensor_builder_set);
  UNUSED_RELEASE(_executor_map);
}

void KernelGenerator::visit(const ir::OpSequence &)
{
  // TODO Implement
}

void KernelGenerator::visit(const ir::operation::While &)
{
  // TODO Implement
}

} // namespace controlflow
} // namespace backend
} // namespace onert
