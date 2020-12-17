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

#include "Conv.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"
#include "ConvPoolHelpers.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/ReshapeOp.h"

namespace mir_onnx
{

void convertConvV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() >= 2);
  auto input = inputs[0];
  auto kernel = inputs[1];

  auto input_shape = input->getShape();
  bool conv1d = false;
  if (input_shape.rank() == 3)
  {
    input_shape = {input_shape.dim(0), input_shape.dim(1), input_shape.dim(2), 1};
    auto reshaped_input = createOp<mir::ops::ReshapeOp>(graph, input, input_shape);
    input = reshaped_input->getOutput(0);
    conv1d = true;
  }
  else
  {
    if (input_shape.rank() != 4)
      throw std::runtime_error{"Conv is unsupported for tensors with more than 4 dimentions"};
  }

  constexpr int num_spatial_dims = 2;

  std::vector<int32_t> dilations(num_spatial_dims, 1);
  if (const auto *dilations_attr = findAttribute(onnx_node, "dilations"))
  {
    dilations = getAttributeValue<std::vector<int32_t>>(*dilations_attr);
    if (conv1d)
      dilations.emplace_back(1);
  }

  if (dilations.size() != num_spatial_dims)
    throw std::runtime_error("Conv: attribute 'dilations' has incorrect size.");
  if (!std::all_of(dilations.cbegin(), dilations.cend(), [](std::int32_t x) { return x == 1; }))
    throw std::runtime_error("Conv: attribute 'dilations' has unsupported value.");

  std::vector<int32_t> strides(num_spatial_dims, 1);
  if (const auto *strides_attr = findAttribute(onnx_node, "strides"))
  {
    strides = getAttributeValue<std::vector<int32_t>>(*strides_attr);
    if (conv1d)
      strides.emplace_back(1);
  }

  if (strides.size() != num_spatial_dims)
    throw std::runtime_error("Conv: attribute 'strides' has incorrect size.");

  // Assuming kernel has OIHW format.
  if (conv1d)
  {
    auto kernel_shape = kernel->getShape();
    assert(kernel_shape.rank() == 3);
    kernel_shape = {kernel_shape.dim(0), kernel_shape.dim(1), kernel_shape.dim(2), 1};
    auto reshaped_kernel = createOp<mir::ops::ReshapeOp>(graph, kernel, kernel_shape);
    kernel = reshaped_kernel->getOutput(0);
  }

  std::vector<std::int32_t> kernel_shape{kernel->getShape().dim(2), kernel->getShape().dim(3)};
  if (const auto *k_shape_attr = findAttribute(onnx_node, "kernel_shape"))
  {
    kernel_shape = getAttributeValue<std::vector<std::int32_t>>(*k_shape_attr);
    if (conv1d)
      kernel_shape.emplace_back(1);
  }

  if (kernel_shape.size() != num_spatial_dims)
    throw std::runtime_error("Conv: attribute 'kernel_shape' has incorrect size.");

  std::vector<std::int32_t> padding_before(num_spatial_dims, 0);
  std::vector<std::int32_t> padding_after(num_spatial_dims, 0);
  if (const auto *pads_attr = findAttribute(onnx_node, "pads"))
  {
    auto pads = getAttributeValue<std::vector<std::int32_t>>(*pads_attr);
    if (conv1d)
    {
      pads.emplace_back(0);
      pads.emplace_back(0);
    }

    if (pads.size() != num_spatial_dims * 2)
      throw std::runtime_error("Conv: attribute 'pads' has incorrect size.");
    const auto fixed_pads = fixPads(input_shape, pads, strides, dilations, kernel_shape);
    padding_before.assign(fixed_pads.cbegin(), std::next(fixed_pads.cbegin(), num_spatial_dims));
    padding_after.assign(std::next(fixed_pads.cbegin(), num_spatial_dims), fixed_pads.cend());
  }
  else
  {
    const auto auto_pad = getAttributeValue<std::string>(onnx_node, "auto_pad", "NOTSET");
    inferAutoPadding(auto_pad, input_shape, dilations, strides, kernel_shape, padding_before,
                     padding_after);
  }

  const auto group = getAttributeValue<std::int64_t>(onnx_node, "group", 1);

  mir::Conv2DOpAttributes attributes;
  attributes.strides = strides;
  attributes.padding_before = padding_before;
  attributes.padding_after = padding_after;
  attributes.num_groups = group;
  attributes.data_format = mir::DataFormat::NCHW;

  std::vector<std::size_t> perm{0, 2, 3, 1}; // OIHW -> OHWI
  kernel = createOp<mir::ops::TransposeOp>(graph, kernel, perm)->getOutput(0);
  auto result = createOp<mir::ops::Conv2DOp>(graph, input, kernel, attributes)->getOutput(0);

  if (inputs.size() > 2)
  {
    auto bias = inputs[2];
    bias = createOp<mir::ops::ReshapeOp>(graph, bias, mir::Shape{1, bias->getShape().dim(0), 1, 1})
             ->getOutput(0);
    result = createOp<mir::ops::AddOp>(graph, result, bias)->getOutput(0);
  }

  if (conv1d)
  {
    auto output_shape = result->getShape();
    output_shape.resize(output_shape.rank() - 1);
    result = createOp<mir::ops::ReshapeOp>(graph, result, output_shape)->getOutput(0);
  }

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
