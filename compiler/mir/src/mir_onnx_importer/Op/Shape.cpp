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

#include "Shape.h"

#include "ONNXHelpers.h"

#include "mir/TensorVariant.h"

#include "mir/ops/ConstantOp.h"

namespace mir_onnx
{

void convertShapeV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  const auto &input_shape = inputs[0]->getShape();
  int size = input_shape.rank();
  mir::Shape output_shape{size};
  std::vector<int64_t> data(static_cast<std::size_t>(size));
  for (int i = 0; i < size; i++)
  {
    data[i] = input_shape.dim(i);
  }
  mir::TensorVariant tensor({mir::DataType::INT64, output_shape}, data.data());
  auto result = createOp<mir::ops::ConstantOp>(graph, tensor)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
