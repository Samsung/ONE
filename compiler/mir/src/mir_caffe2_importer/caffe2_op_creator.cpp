/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "caffe2_op_creator.h"
#include "caffe2_proto_helper.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/CappedReluOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/ResizeOp.h"
#include "mir/ops/SigmoidOp.h"
#include "mir/ops/SoftmaxOp.h"
#include "mir/ops/TransposeOp.h"

#include "mir/Index.h"
#include "mir/Shape.h"
#include "mir/ShapeRange.h"
#include "mir/Tensor.h"
#include "mir/TensorUtil.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace mir_caffe2
{

using namespace ::caffe2;
using namespace mir;

//
// Helper functions
//

static std::pair<std::vector<int32_t>, std::vector<int32_t>>
getPadding(const ::caffe2::OperatorDef &op)
{

  if (hasArgument(op.arg(), "pads"))
  {
    // pads order: t l b r
    auto pads_arg = findArgumentByName(op.arg(), "pads");

    std::vector<int32_t> paddings;
    for (const auto &pad : pads_arg.ints())
      paddings.push_back(static_cast<int32_t>(pad));

    assert(paddings.size() == 4);

    int32_t pad_t = paddings[0];
    int32_t pad_l = paddings[1];
    int32_t pad_b = paddings[2];
    int32_t pad_r = paddings[3];

    std::vector<int32_t> padding_before{pad_t, pad_l};
    std::vector<int32_t> padding_after{pad_b, pad_r};
    return {padding_before, padding_after};
  }

  bool has_custom_pad = hasArgument(op.arg(), "pad_l") || hasArgument(op.arg(), "pad_r") ||
                        hasArgument(op.arg(), "pad_t") || hasArgument(op.arg(), "pad_b");

  if (has_custom_pad)
  {
    int32_t pad_l = getSingleArgument(op, "pad_l", 0);
    int32_t pad_t = getSingleArgument(op, "pad_t", 0);
    int32_t pad_r = getSingleArgument(op, "pad_r", 0);
    int32_t pad_b = getSingleArgument(op, "pad_b", 0);

    std::vector<int32_t> padding_before{pad_t, pad_l};
    std::vector<int32_t> padding_after{pad_b, pad_r};
    return {padding_before, padding_after};
  }

  int32_t pad = getSingleArgument(op, "pad", 0);
  return {{pad, pad}, {pad, pad}};
}

static std::vector<std::int32_t> getStrides(const ::caffe2::OperatorDef &op)
{
  std::vector<std::int32_t> strides;

  if (hasArgument(op.arg(), "stride"))
  {
    std::int32_t stride = getSingleArgument(op, "stride", 1);
    strides = {stride, stride};
  }

  if (hasArgument(op.arg(), "strides"))
  {
    // strides order: h w
    auto strides_arg = findArgumentByName(op.arg(), "strides");
    for (const auto &s : strides_arg.ints())
      strides.push_back(s);
  }

  assert(!strides.empty() && "Strides not found");

  return strides;
}

static std::vector<std::int32_t> getWindowSize(const ::caffe2::OperatorDef &op,
                                               const std::vector<mir::Operation::Output *> &inputs)
{
  int is_global_pooling = getSingleArgument(op, "global_pooling", 0);
  bool has_custom_kernel_size =
    hasArgument(op.arg(), "kernel_h") || hasArgument(op.arg(), "kernel_w");
  bool has_custom_kernels_size = hasArgument(op.arg(), "kernels");

  int kernel_h(0), kernel_w(0);
  if (is_global_pooling)
  {
    const auto &input_shape = inputs[0]->getShape();
    assert(input_shape.rank() == 4 && "getWindowSize() inputs must be of rank 4");
    kernel_h = input_shape.dim(2);
    kernel_w = input_shape.dim(3);
  }
  else
  {
    if (has_custom_kernel_size)
    {
      kernel_h = getSingleArgument(op, "kernel_h", 0);
      kernel_w = getSingleArgument(op, "kernel_w", 0);
    }
    else
    {
      if (has_custom_kernels_size)
      {
        // kernels order: h w
        std::vector<int32_t> kernels;
        auto kernels_arg = findArgumentByName(op.arg(), "kernels");
        for (const auto &ker : kernels_arg.ints())
          kernels.push_back(static_cast<int32_t>(ker));
        assert(kernels.size() == 2);
        kernel_h = kernels[0];
        kernel_w = kernels[1];
      }
      else
      {
        kernel_h = kernel_w = getSingleArgument(op, "kernel", 0);
      }
    }
  }
  return {kernel_h, kernel_w};
}

//
// Check functions
//

static void checkLayout(const OperatorDef &op)
{
  if (getSingleArgument(op, "order", "NCHW") != "NCHW")
    throw std::runtime_error(op.type() + ": only 'NCHW' axis order is supported");
}

static void checkConvLikeOp(const ::caffe2::OperatorDef &op)
{
  checkLayout(op);

  // Padding
  bool has_custom_pad = hasArgument(op.arg(), "pad_l") || hasArgument(op.arg(), "pad_r") ||
                        hasArgument(op.arg(), "pad_t") || hasArgument(op.arg(), "pad_b");

  if (has_custom_pad && hasArgument(op.arg(), "pad"))
    throw std::runtime_error("Custom pad can't be combined with overall pad");

  if (has_custom_pad && !(hasArgument(op.arg(), "pad_l") && hasArgument(op.arg(), "pad_r") &&
                          hasArgument(op.arg(), "pad_t") && hasArgument(op.arg(), "pad_b")))
    throw std::runtime_error("If one custom pad specified - all custom pads must be specified");

  // Kernel size
  bool has_custom_kernel_size =
    hasArgument(op.arg(), "kernel_h") || hasArgument(op.arg(), "kernel_w");

  if (has_custom_kernel_size && hasArgument(op.arg(), "kernel"))
    throw std::runtime_error("Custom kernel size can't be combined with overall kernel size");

  if (has_custom_kernel_size &&
      !(hasArgument(op.arg(), "kernel_h") && hasArgument(op.arg(), "kernel_w")))
    throw std::runtime_error(
      "If one custom kernel size specified - all custom kernel sizes must be specified");
}

static mir::TensorVariant createTensor(const OperatorDef &op)
{
  assert(hasArgument(op.arg(), "shape") && hasArgument(op.arg(), "values"));

  const auto &shape = findArgumentByName(op.arg(), "shape");
  const auto &values = findArgumentByName(op.arg(), "values");

  mir::DataType element_type;
  const void *src_data;
  // if values on floats
  if (!values.floats().empty())
  {
    element_type = mir::DataType::FLOAT32;
    src_data = values.floats().data();
  }
  else
  {
    assert(!values.ints().empty());
    if (op.type() == "GivenTensorInt64Fill")
    {
      element_type = mir::DataType::INT64;
    }
    else
    {
      element_type = mir::DataType::INT32;
    }
    src_data = values.ints().data();
  }

  mir::Shape tensor_shape(shape.ints_size());

  for (int i = 0; i < shape.ints_size(); ++i)
  {
    tensor_shape.dim(i) = shape.ints(i);
  }

  return mir::TensorVariant({element_type, tensor_shape}, src_data);
}

//
// Convert functions
//

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertConstant(const std::vector<mir::Operation::Output *> &,
                                 const ::caffe2::OperatorDef &op)
{
  // Constant may not contain any data if it is a fake input.
  if (!hasArgument(op.arg(), "values"))
    return {};

  return {createOp<ops::ConstantOp>(createTensor(op))->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertAdd(const std::vector<mir::Operation::Output *> &inputs,
                            const ::caffe2::OperatorDef &op)
{
  assert(inputs.size() == 2);
  auto lhs = inputs[0];
  auto rhs = inputs[1];

  if (getSingleArgument(op, "broadcast", 0) != 0)
  {
    // FIXME This only works when 'axis' == 1 and the second input is 1-D.
    rhs = createOp<ops::ReshapeOp>(rhs, Shape{1, rhs->getShape().dim(0), 1, 1})->getOutput(0);
    auto result = createOp<ops::AddOp>(lhs, rhs)->getOutput(0);
    return {result};
  }

  auto result = createOp<ops::AddOp>(lhs, rhs)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertAveragePool(const std::vector<mir::Operation::Output *> &inputs,
                                    const OperatorDef &op)
{
  checkConvLikeOp(op);

  assert(inputs.size() == 1);
  auto input = inputs[0];

  AvgPool2DOpAttributes attributes;
  std::tie(attributes.padding_before, attributes.padding_after) = getPadding(op);
  attributes.window = getWindowSize(op, inputs);
  attributes.strides = getStrides(op);
  attributes.include_pad = false;
  attributes.data_format = DataFormat::NCHW;
  auto result = createOp<ops::AvgPool2DOp>(input, attributes)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertConv(const std::vector<mir::Operation::Output *> &inputs,
                             const ::caffe2::OperatorDef &op)
{
  // dilation order: h w (not used)
  mir::Conv2DOpAttributes attributes;
  attributes.strides = getStrides(op);
  std::tie(attributes.padding_before, attributes.padding_after) = getPadding(op);
  attributes.num_groups = getSingleArgument(op, "group", 1);
  attributes.data_format = DataFormat::NCHW;

  std::vector<std::size_t> perm{0, 2, 3, 1}; // OIHW -> OHWI
  auto kernel = createOp<ops::TransposeOp>(inputs[1], perm)->getOutput(0);
  auto result = createOp<ops::Conv2DOp>(inputs[0], kernel, attributes)->getOutput(0);

  if (op.input_size() > 2)
  {
    auto bias = inputs[2];
    bias = createOp<ops::ReshapeOp>(bias, Shape{1, bias->getShape().dim(0), 1, 1})->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertConcat(const std::vector<mir::Operation::Output *> &inputs,
                               const ::caffe2::OperatorDef &op)
{
  checkLayout(op);

  // `1` corresponds to the default (channels) axis.
  int axis = getSingleArgument(op, "axis", 1);
  auto result = createOp<ops::ConcatOp>(inputs, axis);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertDropout(const std::vector<mir::Operation::Output *> &inputs,
                                const ::caffe2::OperatorDef &)
{
  // This is a no-op in inference mode.
  return {inputs[0]};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertFC(const std::vector<mir::Operation::Output *> &inputs,
                           const ::caffe2::OperatorDef &op)
{
  for (auto &s : {"axis", "axis_w", "float16_compute"})
    if (hasArgument(op.arg(), s))
      throw std::runtime_error(std::string("FC: only default '") + s + "' value is supported");

  const auto &input_shape = inputs[0]->getShape();
  // Transform input into 2-D tensor by flattening axes
  Shape shape{input_shape.dim(0), input_shape.numElements() / input_shape.dim(0)};

  auto reshape = createOp<ops::ReshapeOp>(inputs[0], shape)->getOutput(0);
  auto weights =
    createOp<ops::TransposeOp>(inputs[1], std::vector<std::size_t>{1, 0})->getOutput(0);
  auto result = createOp<ops::FullyConnectedOp>(reshape, weights)->getOutput(0);
  result = createOp<ops::AddOp>(result, inputs[2])->getOutput(0);

  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertMaxPool(const std::vector<mir::Operation::Output *> &inputs,
                                const OperatorDef &op)
{
  checkConvLikeOp(op);

  assert(inputs.size() == 1);
  auto input = inputs[0];

  MaxPool2DOpAttributes attributes;
  std::tie(attributes.padding_before, attributes.padding_after) = getPadding(op);
  attributes.window = getWindowSize(op, inputs);
  attributes.strides = getStrides(op);
  attributes.data_format = DataFormat::NCHW;
  auto result = createOp<ops::MaxPool2DOp>(input, attributes)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertMul(const std::vector<mir::Operation::Output *> &inputs,
                            const ::caffe2::OperatorDef &op)
{
  assert(inputs.size() == 2);
  auto lhs = inputs[0];
  auto rhs = inputs[1];

  if (getSingleArgument(op, "broadcast", 0) != 0)
  {
    // FIXME This only works when `axis` == 1 and the second input is 1-D.
    rhs = createOp<ops::ReshapeOp>(rhs, Shape{1, rhs->getShape().dim(0), 1, 1})->getOutput(0);
    auto result = createOp<ops::MulOp>(lhs, rhs)->getOutput(0);
    return {result};
  }

  auto result = createOp<ops::MulOp>(lhs, rhs)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertRelu(const std::vector<mir::Operation::Output *> &inputs)
{
  auto relu = createOp<ops::ReluOp>(inputs[0]);
  return {relu->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertResizeNearest(const std::vector<mir::Operation::Output *> &inputs,
                                      const ::caffe2::OperatorDef &op)
{
  std::vector<float> scales(4);
  assert(inputs[0]->getShape().rank() == 4 && "only 4d tensors is supported");
  // Assuming NCHW format.
  scales[0] = 1.0f;
  scales[1] = 1.0f;
  scales[2] = getSingleArgument(op, "height_scale", 1.0f);
  scales[3] = getSingleArgument(op, "width_scale", 1.0f);
  auto result =
    createOp<ops::ResizeOp>(inputs[0], ops::ResizeOp::ResizeMethod::nearestNeighbor, scales)
      ->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertSigmoid(const std::vector<mir::Operation::Output *> &inputs)
{
  auto result = createOp<ops::SigmoidOp>(inputs[0]);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertSoftmax(const std::vector<mir::Operation::Output *> &inputs,
                                const ::caffe2::OperatorDef &op)
{
  int axis = getSingleArgument(op, "axis", 1);
  auto softmax = createOp<ops::SoftmaxOp>(inputs[0], axis);
  return {softmax->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertSpatialBN(const std::vector<mir::Operation::Output *> &inputs,
                                  const ::caffe2::OperatorDef &op)
{
  checkLayout(op);

  // Sanity checks
  if (op.input_size() != 5)
    throw std::runtime_error(
      "SpatialBN must have exactly 5 inputs ('sums' and 'sumsq' are not supported yet)");
  if (getSingleArgument(op, "is_test", 1) != 1)
    throw std::runtime_error("SpatialBN: only test mode supported");

  // overall_res = (X - mean) / sqrt(var + epsilon) * scale + bias

  auto scale_op = dynamic_cast<mir::ops::ConstantOp *>(inputs[1]->getNode());
  auto bias_op = dynamic_cast<mir::ops::ConstantOp *>(inputs[2]->getNode());
  auto mean_op = dynamic_cast<mir::ops::ConstantOp *>(inputs[3]->getNode());
  auto var_op = dynamic_cast<mir::ops::ConstantOp *>(inputs[4]->getNode());
  if (scale_op == nullptr || bias_op == nullptr || mean_op == nullptr || var_op == nullptr)
    throw std::runtime_error(
      "SpatialBN: non-constant 'scale', 'bias', 'mean' and 'var' inputs are not supported yet.");

  const auto &scale_tensor = scale_op->getValue();
  const auto &bias_tensor = bias_op->getValue();
  const auto &mean_tensor = mean_op->getValue();
  const auto &var_tensor = var_op->getValue();
  float eps = getSingleArgument(op, "epsilon", 1e-5f);

  // res1 = X - mean
  Tensor<float> bias_data(mean_tensor);
  for (auto &idx : ShapeRange(bias_data.getShape()))
    bias_data.at(idx) *= -1;

  auto mean = createOp<ops::ConstantOp>(mean_tensor)->getOutput(0);
  mean = createOp<ops::ReshapeOp>(mean, Shape{1, mean->getShape().dim(0), 1, 1})->getOutput(0);
  auto result = createOp<ops::AddOp>(inputs[0], mean)->getOutput(0);

  // res2 = res1 * scale / (var + epsilon)
  Tensor<float> multiplier(scale_tensor);
  for (auto &idx : ShapeRange(scale_tensor.getShape()))
    multiplier.at(idx) /= std::sqrt(*reinterpret_cast<float *>(var_tensor.at(idx)) + eps);
  auto scale = createOp<ops::ConstantOp>(scale_tensor)->getOutput(0);
  scale = createOp<ops::ReshapeOp>(scale, Shape{1, scale->getShape().dim(0), 1, 1})->getOutput(0);
  result = createOp<ops::MulOp>(result, scale)->getOutput(0);

  // overall_res = res2 + bias
  auto bias = createOp<ops::ConstantOp>(bias_tensor)->getOutput(0);
  bias = createOp<ops::ReshapeOp>(bias, Shape{1, bias->getShape().dim(0), 1, 1})->getOutput(0);
  result = createOp<ops::AddOp>(result, bias)->getOutput(0);

  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertSum(const std::vector<mir::Operation::Output *> &inputs)
{
  auto result = createOp<ops::AddOp>(inputs[0], inputs[1])->getOutput(0);
  for (int i = 2; i < static_cast<int>(inputs.size()); ++i)
  {
    result = createOp<ops::AddOp>(result, inputs[i])->getOutput(0);
  }
  return {result};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertClip(const std::vector<mir::Operation::Output *> &inputs,
                             const ::caffe2::OperatorDef &op)
{

  float max = getSingleArgument(op, "max", float(0));
  float min = getSingleArgument(op, "min", float(0));

  if (min != 0.0f)
    throw std::runtime_error("Clip: min != 0 is not supported.");
  if (max <= min)
    throw std::runtime_error("Clip: max <= min is not supported.");
  auto cap_relu = createOp<ops::CappedReluOp>(inputs[0], max);

  return {cap_relu->getOutput(0)};
}

std::vector<mir::Operation::Output *>
Caffe2OpCreator::convertReshape(const std::vector<mir::Operation::Output *> &inputs,
                                const ::caffe2::OperatorDef &)
{
  auto shape_op = dynamic_cast<mir::ops::ConstantOp *>(inputs[1]->getNode());
  if (shape_op == nullptr)
    throw std::runtime_error("Reshape: non-constant shape is not supported yet.");

  const auto &shape_tensor = shape_op->getValue();

  Tensor<int64_t> out_shape_tensor(shape_tensor);

  ShapeRange range(out_shape_tensor.getShape());
  std::vector<int32_t> shape_vec;
  for (const auto &index : range)
  {
    shape_vec.push_back(static_cast<int32_t>(out_shape_tensor.at(index)));
  }
  Shape out_shape(shape_vec);

  auto reshape = createOp<ops::ReshapeOp>(inputs[0], out_shape);

  return {reshape->getOutput(0)};
}

} // namespace mir_caffe2
