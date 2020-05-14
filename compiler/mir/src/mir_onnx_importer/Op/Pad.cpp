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

#include "Pad.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/PadOp.h"

namespace mir_onnx
{

void convertPadAttrName(const std::string &pad_attr_name, const onnx::NodeProto &onnx_node,
                        ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  // 0.0f is the default value to be filled into padded cells.
  const auto value = getAttributeValue<float>(onnx_node, "value", 0.0f);
  const auto pads = getAttributeValue<std::vector<std::int64_t>>(onnx_node, pad_attr_name);
  // "constant" is the default mode.
  const auto mode = getAttributeValue<std::string>(onnx_node, "mode", "constant");
  if (mode != "constant")
    throw std::runtime_error("Not supported Pad mode attribute!");

  const int num_dims = input->getShape().rank();
  assert(static_cast<int>(pads.size()) == num_dims * 2);
  mir::PadOpAttributes attributes(num_dims);
  for (int i = 0; i < num_dims; i++)
  {
    attributes.padding_before[i] = pads[i];
    attributes.padding_after[i] = pads[num_dims + i];
  }

  attributes.padding_value = value;

  auto result = createOp<mir::ops::PadOp>(graph, input, attributes)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertPadV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertPadAttrName("paddings", onnx_node, context);
}

void convertPadV2(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertPadAttrName("pads", onnx_node, context);
}

} // namespace mir_onnx
