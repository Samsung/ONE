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

void StaticInferer::visit(const ir::operation::If &op)
{
  for (const auto input_idx : op.getInputs())
  {
    if (_operands.at(input_idx).info().isDynamic())
    {
      for (const auto output_idx : op.getOutputs())
      {
        _operands.at(output_idx).info().setDynamic();
      }
      return;
    }
  }
  // If operation cannot infer shape of outputs without outputs of child subgraph
}

} // namespace shape_inference
} // namespace onert
