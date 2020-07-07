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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::Permute &op)
{
  const auto input_idx{op.getInputs().at(0)};
  const auto &input = _operands.at(input_idx);
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  // Permute is a special operation that layouts of input/output may be different on backend
  // However, it is not applied here, so input/output have the same layout of frontend. Because
  // "ExecutorFactory" would convert shape of input/output accoding to the layouts when registering
  // operand info to "TensorBuilder" after calling "StaticInferer"
  const auto new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Permute & /* op */)
{
  // NOTE Permute is a special operation which does not do shape inference before the actual
  // function(kernel) execution. Shape inference and output allocation will be done in the kernel
  // on-the-fly, as it must support inter-backend inference/allocation.
}

} // namespace shape_inference
} // namespace onert
