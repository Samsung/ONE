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

#include "MaxPool.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"
#include "ConvPoolHelpers.h"

#include "mir/ops/MaxPool2DOp.h"

namespace mir_onnx
{

void convertMaxPoolV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  const auto &input_shape = input->getShape();
  if (input_shape.rank() != 4)
    throw std::runtime_error("MaxPool: only 2-D input is supported.");

  constexpr int num_spatial_dims = 2;

  const auto strides =
    getAttributeValue(onnx_node, "strides", std::vector<std::int32_t>(num_spatial_dims, 1));
  if (strides.size() != num_spatial_dims)
    throw std::runtime_error("MaxPool: attribute 'strides' has incorrect size.");

  const auto kernel_shape = getAttributeValue<std::vector<std::int32_t>>(onnx_node, "kernel_shape");
  if (kernel_shape.size() != num_spatial_dims)
    throw std::runtime_error("MaxPool: attribute 'kernel_shape' has incorrect size.");

  std::vector<std::int32_t> padding_before;
  std::vector<std::int32_t> padding_after;
  if (const auto *pads_attr = findAttribute(onnx_node, "pads"))
  {
    const auto pads = getAttributeValue<std::vector<std::int32_t>>(*pads_attr);
    if (pads.size() != num_spatial_dims * 2)
      throw std::runtime_error("MaxPool: attribute 'pads' has incorrect size.");
    padding_before.assign(pads.cbegin(), std::next(pads.cbegin(), num_spatial_dims));
    padding_after.assign(std::next(pads.cbegin(), num_spatial_dims), pads.cend());
  }
  else
  {
    const auto auto_pad = getAttributeValue<std::string>(onnx_node, "auto_pad", "NOTSET");
    const std::vector<std::int32_t> dilations(num_spatial_dims, 1);
    inferAutoPadding(auto_pad, input_shape, dilations, strides, kernel_shape, padding_before,
                     padding_after);
  }

  mir::MaxPool2DOpAttributes attributes;
  attributes.window = kernel_shape;
  attributes.strides = strides;
  attributes.padding_before = padding_before;
  attributes.padding_after = padding_after;
  attributes.data_format = mir::DataFormat::NCHW;
  auto result = createOp<mir::ops::MaxPool2DOp>(graph, input, attributes)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertMaxPoolV8(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto storage_order = getAttributeValue<int64_t>(onnx_node, "storage_order", 0);
  if (storage_order != 0)
    throw std::runtime_error("Not supported storage order attribute!");

  convertMaxPoolV1(onnx_node, context);
}

void convertMaxPoolV10(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto ceil_mode = getAttributeValue<int64_t>(onnx_node, "ceil_mode", 0);
  if (ceil_mode != 0)
    throw std::runtime_error("Not supported ceil_mode attribute!");

  const auto *dilations = findAttribute(onnx_node, "dilations");
  if (dilations != nullptr)
  {
    // check default (=1) dilations on each spatial axis
    for (auto index = 0; index < dilations->ints_size(); index++)
      if (dilations->ints(index) != 1)
        throw std::runtime_error("Not supported dilations in MaxPool operation!");
  }

  convertMaxPoolV8(onnx_node, context);
}

} // namespace mir_onnx
