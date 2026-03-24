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

#include "ConvTranspose.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"
#include "ConvPoolHelpers.h"

#include "mir/TensorUtil.h"
#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/ReshapeOp.h"

namespace mir_onnx
{

void convertConvTransposeV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() >= 2);
  auto input = inputs[0];
  auto kernel = inputs[1];

  const auto group = getAttributeValue<std::int64_t>(onnx_node, "group", 1);
  if (group != 1)
    throw std::runtime_error("ConvTranspose: attribute 'group' has unsupported value.");

  const auto &input_shape = input->getShape();
  if (input_shape.rank() != 4)
    throw std::runtime_error("ConvTranspose: only 2-D input is supported.");

  constexpr int num_spatial_dims = 2;

  const auto dilations =
    getAttributeValue(onnx_node, "dilations", std::vector<std::int32_t>(num_spatial_dims, 1));
  if (dilations.size() != num_spatial_dims)
    throw std::runtime_error("ConvTranspose: attribute 'dilations' has incorrect size.");
  if (!std::all_of(dilations.cbegin(), dilations.cend(), [](std::int32_t x) { return x == 1; }))
    throw std::runtime_error("ConvTranspose: attribute 'dilations' has unsupported value.");

  const auto strides =
    getAttributeValue(onnx_node, "strides", std::vector<std::int32_t>(num_spatial_dims, 1));
  if (strides.size() != num_spatial_dims)
    throw std::runtime_error("ConvTranspose: attribute 'strides' has incorrect size.");

  const auto output_padding =
    getAttributeValue(onnx_node, "output_padding", std::vector<std::int32_t>(num_spatial_dims, 0));
  if (output_padding.size() != num_spatial_dims)
    throw std::runtime_error("ConvTranspose: attribute 'output_padding' has incorrect size.");
  if (!std::all_of(output_padding.cbegin(), output_padding.cend(),
                   [](std::int32_t x) { return x == 0; }))
    throw std::runtime_error("ConvTranspose: attribute 'output_padding' has unsupported value.");

  // Assuming kernel has IOHW format.
  assert(kernel->getShape().rank() == 4);
  const auto kernel_size = getAttributeValue(
    onnx_node, "kernel_shape",
    std::vector<std::int32_t>{kernel->getShape().dim(2), kernel->getShape().dim(3)});
  if (kernel_size.size() != num_spatial_dims)
    throw std::runtime_error("ConvTranspose: attribute 'kernel_shape' has incorrect size.");

  // ONNX IOHW -> MIR HWOI
  std::vector<std::size_t> perm{2, 3, 1, 0}; // OIHW -> OHWI
  kernel = createOp<mir::ops::TransposeOp>(graph, kernel, perm)->getOutput(0);

  mir::Operation::Output *result;
  if (const auto *output_shape_attr = findAttribute(onnx_node, "output_shape"))
  {
    const auto output_size = getAttributeValue<std::vector<std::int32_t>>(*output_shape_attr);
    if (output_size.size() != num_spatial_dims)
      throw std::runtime_error("ConvTranspose: attribute 'output_shape' has incorrect size.");
    const mir::Shape output_shape{input_shape.dim(0), kernel->getShape().dim(2), output_size[0],
                                  output_size[1]};
    mir::Deconv2DOpAttributes attributes;
    attributes.strides = strides;
    attributes.data_format = mir::DataFormat::NCHW;
    attributes.padding_type = mir::ops::PaddingType::SameUpper;
    result =
      createOp<mir::ops::DeConv2DOp>(graph, input, kernel, attributes, output_shape)->getOutput(0);
  }
  else
  {
    // TODO This code was not tested.
    throw std::runtime_error(
      "ConvTranspose: absence of attribute 'output_shape' is not supported.");
    std::vector<std::int32_t> padding_before(num_spatial_dims, 0);
    std::vector<std::int32_t> padding_after(num_spatial_dims, 0);
    if (const auto *pads_attr = findAttribute(onnx_node, "pads"))
    {
      const auto pads = getAttributeValue<std::vector<std::int32_t>>(*pads_attr);
      if (pads.size() != num_spatial_dims * 2)
        throw std::runtime_error("ConvTranspose: attribute 'pads' has incorrect size.");
      padding_before.assign(pads.cbegin(), std::next(pads.cbegin(), num_spatial_dims));
      padding_after.assign(std::next(pads.cbegin(), num_spatial_dims), pads.cend());
    }
    else
    {
      const auto auto_pad = getAttributeValue<std::string>(onnx_node, "auto_pad", "NOTSET");
      inferAutoPadding(auto_pad, input_shape, dilations, strides, kernel_size, padding_before,
                       padding_after);
    }
    mir::Deconv2DOpAttributes attributes;
    attributes.strides = strides;
    attributes.padding_before = padding_before;
    attributes.padding_after = padding_after;
    attributes.data_format = mir::DataFormat::NCHW;
    result = createOp<mir::ops::DeConv2DOp>(graph, input, kernel, attributes)->getOutput(0);
  }

  if (inputs.size() > 2)
  {
    auto bias = inputs[2];
    bias = createOp<mir::ops::ReshapeOp>(graph, bias, mir::Shape{1, bias->getShape().dim(0), 1, 1})
             ->getOutput(0);
    result = createOp<mir::ops::AddOp>(graph, result, bias)->getOutput(0);
  }

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
