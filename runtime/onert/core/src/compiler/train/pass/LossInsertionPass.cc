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

#include "LossInsertionPass.h"

namespace onert
{
namespace compiler
{
namespace train
{
namespace pass
{

void LossInsertionPass::run()
{
  // TODO Check if it is necessary to add Loss op to this graph

  // TODO Find the loss information of this graph

  // TODO Find the correct pred_index and add the y_true operand into this graph if necessary

  // TODO Set inputs of loss op

  // TODO Find the correct Shape and TypeInfo of the output of loss op
  // TODO Add the output operand into this graph

  // TODO Set outputs

  // TODO Add ir::train::operation::Loss with the correct inputs, outputs, and param

  // TODO Change outputs of graph if necessary
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
