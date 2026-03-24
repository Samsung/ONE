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

#include "Reshape.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/Tensor.h"
#include "mir/ShapeRange.h"

#include "mir/ops/ConstantOp.h"
#include "mir/ops/ReshapeOp.h"

namespace mir_onnx
{

void convertReshapeV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  // consumed_inputs attribute not used
  const auto *shape_attr = findAttribute(onnx_node, "shape");
  if (shape_attr && shape_attr->ints_size() > 0)
  {
    mir::Shape in_shape = inputs[0]->getShape();
    mir::Shape out_shape(shape_attr->ints_size());
    for (int32_t index = 0; index < out_shape.rank(); index++)
    {
      const auto dim_value = shape_attr->ints(index);
      if (dim_value == 0)
        out_shape.dim(index) = in_shape.dim(index);
      else
        out_shape.dim(index) = dim_value;
    }

    auto result = createOp<mir::ops::ReshapeOp>(graph, inputs[0], out_shape)->getOutput(0);

    context->setNodeOutputs(onnx_node, {result});
  }
  else // dimension value is unchanged
  {
    context->setNodeOutputs(onnx_node, {inputs[0]});
  }
}

void convertReshapeV5(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  // The original shape
  const auto &in_shape = inputs[0]->getShape();

  // Input tensor describing the new shape
  auto *op = dynamic_cast<mir::ops::ConstantOp *>(inputs[1]->getNode());
  assert(op && "We support only constant shape input");
  auto shape_tensor = op->getValue();
  mir::Shape shape_tensor_shape = (shape_tensor).getShape();
  assert(shape_tensor_shape.rank() == 1);
  // The rank of the new shape
  auto cnt = shape_tensor_shape.numElements();
  // The vector to build the new shape from
  std::vector<int32_t> shape_vector(cnt);
  mir::ShapeRange out_range(shape_tensor_shape);
  mir::Tensor<int64_t> tensor_accessor(shape_tensor);

  int i = 0;
  for (auto idx : out_range)
  {
    if (tensor_accessor.at(idx) == 0)
      shape_vector[i] = in_shape.dim(i);
    else if (tensor_accessor.at(idx) == -1)
      shape_vector[i] = mir::Shape::autoDim;
    else
      shape_vector[i] = tensor_accessor.at(idx);
    i++;
  }
  auto out_shape = mir::Shape(shape_vector);
  auto result = createOp<mir::ops::ReshapeOp>(graph, inputs[0], out_shape)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
