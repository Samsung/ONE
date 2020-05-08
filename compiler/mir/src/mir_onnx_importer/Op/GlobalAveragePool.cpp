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

#include "GlobalAveragePool.h"

#include "ONNXHelpers.h"

#include "mir/ops/AvgPool2DOp.h"

namespace mir_onnx
{

void convertGlobalAveragePoolV2(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  const auto &input_shape = input->getShape();
  if (input_shape.rank() != 4)
    throw std::runtime_error("GlobalAveragePool: only 2-D input is supported.");

  // GlobalAveragePool is equivalent to AveragePool with kernel size equal
  // to the spatial dimension of input tensor.
  const std::vector<std::int32_t> window_size{input->getShape().dim(2), input->getShape().dim(3)};
  mir::AvgPool2DOpAttributes attributes;
  attributes.window = window_size;
  attributes.data_format = mir::DataFormat::NCHW;

  auto result = createOp<mir::ops::AvgPool2DOp>(graph, input, attributes)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
