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

#include "PermutationInsertionPass.h"

namespace onert
{
namespace compiler
{
namespace train
{
namespace pass
{

void PermutationInsertionPass::callback(const ir::OperandIndex &index, ir::Operand &object)
{
  // NOTE Permutation is not inserted for outputs of trainable graph unless user requests result of
  //      output.
  if (_graph.getOutputs().contains(index))
    return;

  compiler::pass::PermutationInsertionPass::callback(index, object);
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
