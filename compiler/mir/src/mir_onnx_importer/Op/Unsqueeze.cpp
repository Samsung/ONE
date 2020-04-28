/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Unsqueeze.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/ReshapeOp.h"

namespace mir_onnx
{

void convertUnsqueezeV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  const auto axes = getAttributeValue<std::vector<std::int64_t>>(onnx_node, "axes");
  assert(!axes.empty());
  const mir::Shape &input_shape = inputs[0]->getShape();
  const int out_rank = input_shape.rank() + static_cast<int>(axes.size());
  mir::Shape out_shape(out_rank);
  auto ints_iterator = axes.cbegin();
  int j = 0;
  for (int i = 0; i < out_rank; i++)
  {
    if (ints_iterator < axes.cend() && i == *ints_iterator)
    {
      out_shape.dim(i) = 1;
      ints_iterator++;
    }
    else
    {
      out_shape.dim(i) = input_shape.dim(j);
      j++;
    }
  }
  auto result = createOp<mir::ops::ReshapeOp>(graph, inputs[0], out_shape)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
