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

#include "tflite_op_creator.h"
#include "schema_generated.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/CappedReluOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/DivOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/HardSwishOp.h"
#include "mir/ops/LeakyReluOp.h"
#include "mir/ops/MaxOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/PadOp.h"
#include "mir/ops/ReduceMeanOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/ResizeOp.h"
#include "mir/ops/SigmoidOp.h"
#include "mir/ops/SliceOp.h"
#include "mir/ops/SoftmaxOp.h"
#include "mir/ops/SqrtOp.h"
#include "mir/ops/SqueezeOp.h"
#include "mir/ops/SubOp.h"
#include "mir/ops/TanhOp.h"
#include "mir/ops/TransposeOp.h"

#include "mir/Shape.h"
#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <stdexcept>

namespace mir_tflite
{

namespace ops = mir::ops;
using mir::Shape;

static mir::ops::PaddingType convertPadding(tflite::Padding padding)
{
  switch (padding)
  {
    case tflite::Padding_VALID:
      return mir::ops::PaddingType::Valid;
    case tflite::Padding_SAME:
      return mir::ops::PaddingType::SameUpper;
    default:
      throw std::runtime_error(std::string("Unsupported Padding: ") +
                               tflite::EnumNamePadding(padding));
  }
}

// TODO Move this to MIR?
static void calculatePadding(mir::ops::PaddingType padding_type, const mir::Shape &input_shape,
                             const std::vector<std::int32_t> &window_size,
                             const std::vector<std::int32_t> &strides,
                             std::vector<std::int32_t> &padding_before,
                             std::vector<std::int32_t> &padding_after)
{
  constexpr int num_spatial_dims = 2;
  assert(window_size.size() == num_spatial_dims);
  assert(strides.size() == num_spatial_dims);
  assert(padding_before.size() == num_spatial_dims);
  assert(padding_after.size() == num_spatial_dims);

  switch (padding_type)
  {
    case mir::ops::PaddingType::SameUpper:
      for (int i = 0; i < num_spatial_dims; ++i)
      {
        // Assuming NHWC format.
        const std::int32_t total_padding =
          (input_shape.dim(1 + i) % strides[i] == 0)
            ? std::max(0, window_size[i] - strides[i])
            : std::max(0, window_size[i] - input_shape.dim(1 + i) % strides[i]);
        padding_before[i] = total_padding / 2;
        padding_after[i] = total_padding - padding_before[i];
      }
      break;
    case mir::ops::PaddingType::Valid:
      for (int i = 0; i < num_spatial_dims; ++i)
      {
        padding_before[i] = 0;
        padding_after[i] = 0;
      }
      break;
    default:
      assert(false);
  }
}

template <typename VectorT>
static std::vector<VectorT> convertIntTensorToVector(const mir::Tensor<int32_t> &tensor)
{
  std::vector<VectorT> v;
  for (const auto &i : mir::ShapeRange(tensor.getShape()))
    v.emplace_back(static_cast<VectorT>(tensor.at(i)));
  return v;
}

static const mir::TensorVariant &extractTensor(const mir::Operation::Output *output)
{
  auto constant_op = dynamic_cast<const ops::ConstantOp *>(output->getNode());
  if (constant_op == nullptr)
    throw std::runtime_error("Non-constant input is not supported.");
  return constant_op->getValue();
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertConv2D(const tflite::Conv2DOptionsT *opts,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  auto kernel = inputs.at(1);
  auto bias = inputs.at(2);

  mir::Conv2DOpAttributes attributes;
  attributes.strides = {opts->stride_h, opts->stride_w};

  const auto padding_type = convertPadding(opts->padding);
  const auto &input_shape = input->getShape();
  const auto &kernel_shape = kernel->getShape();
  const auto &strides = attributes.strides;
  auto &pad_before = attributes.padding_before;
  auto &pad_after = attributes.padding_after;
  std::vector<std::int32_t> kernel_size{kernel_shape.dim(1), kernel_shape.dim(2)};
  calculatePadding(padding_type, input_shape, kernel_size, strides, pad_before, pad_after);

  mir::Operation::Output *result;
  if (input->getType().isQuantized())
  {
    result = createOp<ops::Conv2DOp>(input, kernel, bias, attributes)->getOutput(0);
  }
  else // TODO Fuse bias to other backends
  {
    result = createOp<ops::Conv2DOp>(input, kernel, attributes)->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertDepthwiseConv2D(const tflite::DepthwiseConv2DOptionsT *opts,
                                        const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  auto kernel = inputs.at(1);
  auto bias = inputs.at(2);

  // OHWI -> HWIO
  const std::vector<std::size_t> axis_order{1, 2, 3, 0};
  kernel = createOp<ops::TransposeOp>(kernel, axis_order)->getOutput(0);

  mir::Conv2DOpAttributes attributes;
  attributes.strides = {opts->stride_h, opts->stride_w};

  const auto padding_type = convertPadding(opts->padding);
  const auto &input_shape = input->getShape();
  const auto &kernel_shape = kernel->getShape();
  std::vector<std::int32_t> kernel_size{kernel_shape.dim(0), kernel_shape.dim(1)};
  const auto &strides = attributes.strides;
  auto &pad_before = attributes.padding_before;
  auto &pad_after = attributes.padding_after;
  calculatePadding(padding_type, input_shape, kernel_size, strides, pad_before, pad_after);

  mir::Operation::Output *result;
  if (input->getType().isQuantized())
  {
    result = createOp<ops::DepthwiseConv2DOp>(input, kernel, bias, attributes)->getOutput(0);
  }
  else // TODO Fuse bias to other backends
  {
    result = createOp<ops::DepthwiseConv2DOp>(input, kernel, attributes)->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertConcatenation(const tflite::ConcatenationOptionsT *opts,
                                      const std::vector<mir::Operation::Output *> &inputs)
{
  auto result = createOp<ops::ConcatOp>(inputs, opts->axis);
  return {addFusedActivation(result->getOutput(0), opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertMaxPool2D(const tflite::Pool2DOptionsT *opts,
                                  const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  const auto &input_shape = input->getShape();

  mir::MaxPool2DOpAttributes attributes;
  attributes.window = {opts->filter_height, opts->filter_width};
  attributes.strides = {opts->stride_h, opts->stride_w};

  const auto padding_type = convertPadding(opts->padding);
  const auto &window_size = attributes.window;
  const auto &strides = attributes.strides;
  auto &pad_before = attributes.padding_before;
  auto &pad_after = attributes.padding_after;
  calculatePadding(padding_type, input_shape, window_size, strides, pad_before, pad_after);

  auto result = createOp<ops::MaxPool2DOp>(input, attributes);
  return {addFusedActivation(result->getOutput(0), opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertAveragePool2D(const tflite::Pool2DOptionsT *opts,
                                      const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  const auto &input_shape = input->getShape();

  mir::AvgPool2DOpAttributes attributes;
  attributes.window = {opts->filter_height, opts->filter_width};
  attributes.strides = {opts->stride_h, opts->stride_w};
  attributes.include_pad = false;

  const auto padding_type = convertPadding(opts->padding);
  const auto &window_size = attributes.window;
  const auto &strides = attributes.strides;
  auto &pad_before = attributes.padding_before;
  auto &pad_after = attributes.padding_after;
  calculatePadding(padding_type, input_shape, window_size, strides, pad_before, pad_after);

  auto result = createOp<ops::AvgPool2DOp>(input, attributes);
  return {addFusedActivation(result->getOutput(0), opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSoftmax(const tflite::SoftmaxOptionsT * /*opts*/,
                                const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  // Softmax in TFLite is always 2-D.
  assert(input->getShape().rank() == 2);
  const int32_t axis = 1;
  auto result = createOp<ops::SoftmaxOp>(input, axis);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSlice(const tflite::SliceOptionsT * /*opts*/,
                              const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  mir::Tensor<int32_t> begin_tensor(extractTensor(inputs.at(1)));
  mir::Tensor<int32_t> size_tensor(extractTensor(inputs.at(2)));

  Shape starts(convertIntTensorToVector<int32_t>(begin_tensor));
  Shape sizes(convertIntTensorToVector<int32_t>(size_tensor));
  auto result = createOp<ops::SliceOp>(input, starts, sizes);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertReshape(const tflite::ReshapeOptionsT *opts,
                                const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  // TODO: we should also support "-1" values in new_shape, which means that correct
  // shape values must be calculated. Better do it in the shape inference module.
  Shape new_shape(opts->new_shape.size());
  for (int i = 0; i < static_cast<int>(opts->new_shape.size()); ++i)
  {
    new_shape.dim(i) = opts->new_shape[i];
  }
  auto result = createOp<ops::ReshapeOp>(input, new_shape);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertTransposeConv(const tflite::TransposeConvOptionsT *opts,
                                      const std::vector<mir::Operation::Output *> &inputs)
{
  mir::Tensor<int32_t> output_shape_tensor(extractTensor(inputs.at(0)));
  auto kernel = inputs.at(1);
  auto input = inputs.at(2);

  mir::Deconv2DOpAttributes attributes;
  attributes.strides = {opts->stride_h, opts->stride_w};
  Shape output_shape(convertIntTensorToVector<int32_t>(output_shape_tensor));

  // OHWI -> HWOI
  const std::vector<std::size_t> axis_order{1, 2, 0, 3};
  kernel = createOp<ops::TransposeOp>(kernel, axis_order)->getOutput(0);

  attributes.padding_type = convertPadding(opts->padding);
  auto result = createOp<ops::DeConv2DOp>(input, kernel, attributes, output_shape)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertResizeNearestNeighbor(const tflite::ResizeNearestNeighborOptionsT *opts,
                                              const std::vector<mir::Operation::Output *> &inputs)
{
  if (opts->align_corners)
    throw std::runtime_error("'align_corners' is not currently supported");

  auto input = inputs.at(0);
  mir::Tensor<int32_t> size_tensor(extractTensor(inputs.at(1)));

  const auto &input_shape = input->getShape();
  Shape res_shape{input_shape.dim(0), size_tensor.at(mir::Index{0}), size_tensor.at(mir::Index{1}),
                  input_shape.dim(3)};
  auto result =
    createOp<ops::ResizeOp>(input, ops::ResizeOp::ResizeMethod::nearestNeighbor, res_shape);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertAdd(const tflite::AddOptionsT *opts,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::AddOp>(inputs[0], inputs[1])->getOutput(0);
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSub(const tflite::SubOptionsT *opts,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::SubOp>(inputs[0], inputs[1])->getOutput(0);
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertMul(const tflite::MulOptionsT *opts,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::MulOp>(inputs[0], inputs[1])->getOutput(0);
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertDiv(const tflite::DivOptionsT *opts,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::DivOp>(inputs[0], inputs[1])->getOutput(0);
  return {addFusedActivation(result, opts->fused_activation_function)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertMax(const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::MaxOp>(inputs[0], inputs[1])->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSquaredDifference(const std::vector<mir::Operation::Output *> &inputs)
{
  assert(inputs.size() == 2);
  auto result = createOp<ops::SubOp>(inputs[0], inputs[1])->getOutput(0);
  result = createOp<ops::MulOp>(result, result)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertMean(const tflite::ReducerOptionsT *opts,
                             const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  mir::Tensor<int32_t> axes_tensor(extractTensor(inputs.at(1)));

  std::vector<int32_t> axes = convertIntTensorToVector<int32_t>(axes_tensor);
  auto result = createOp<ops::ReduceMeanOp>(input, axes, opts->keep_dims);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertFullyConnected(const tflite::FullyConnectedOptionsT *opts,
                                       const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  auto weights = inputs.at(1);
  auto bias = inputs.at(2);

  // Flatten input to 2-D shape.
  const auto &input_shape = input->getShape();
  int32_t outer_size = input_shape.dim(0);
  int32_t inner_size = input_shape.numElements() / outer_size;
  auto flatten = createOp<ops::ReshapeOp>(input, Shape{outer_size, inner_size})->getOutput(0);

  // Transpose the weights.
  const std::vector<std::size_t> axis_order{1, 0};
  weights = createOp<ops::TransposeOp>(weights, axis_order)->getOutput(0);

  mir::Operation::Output *result;
  if (input->getType().isQuantized())
  {
    result = createOp<ops::FullyConnectedOp>(flatten, weights, bias)->getOutput(0);
  }
  else // TODO Fuse bias to other backends
  {
    result = createOp<ops::FullyConnectedOp>(flatten, weights)->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }
  return {addFusedActivation(result, opts->fused_activation_function)};
}

mir::Operation::Output *
TFLiteOpCreator::addFusedActivation(mir::Operation::Output *input,
                                    tflite::ActivationFunctionType activation_type)
{
  switch (activation_type)
  {
    case tflite::ActivationFunctionType_NONE:
      return input;
    case tflite::ActivationFunctionType_RELU:
      return createOp<ops::ReluOp>(input)->getOutput(0);
    case tflite::ActivationFunctionType_RELU6:
      return createOp<ops::CappedReluOp>(input, 6)->getOutput(0);
    case tflite::ActivationFunctionType_TANH:
      return createOp<ops::TanhOp>(input)->getOutput(0);
    default:
      throw std::runtime_error(std::string("Unsupported activation type: ") +
                               tflite::EnumNameActivationFunctionType(activation_type));
  }
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSqueeze(const tflite::SqueezeOptionsT *opts,
                                const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  std::vector<int32_t> squeeze_dims(opts->squeeze_dims.begin(), opts->squeeze_dims.end());
  auto result = createOp<ops::SqueezeOp>(input, squeeze_dims);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertPad(const tflite::PadOptionsT * /*opts*/,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  mir::Tensor<int32_t> paddings_tensor(extractTensor(inputs.at(1)));

  const auto &input_shape = input->getShape();
  const int num_dims = input_shape.rank();

  mir::PadOpAttributes attributes(num_dims);
  for (int i = 0; i < num_dims; i++)
  {
    attributes.padding_before[i] = paddings_tensor.at(mir::Index({i, 0}));
    attributes.padding_after[i] = paddings_tensor.at(mir::Index({i, 1}));
  }

  auto result = createOp<ops::PadOp>(input, attributes)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertTanh(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::TanhOp>(input);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertReLU(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::ReluOp>(input);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertReLU6(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::CappedReluOp>(input, 6);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertRsqrt(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  const float one_value = 1.0f;
  mir::TensorVariant one_tensor({mir::DataType::FLOAT32, {}}, &one_value);
  auto one = createOp<ops::ConstantOp>(one_tensor)->getOutput(0);
  auto sqrt = createOp<ops::SqrtOp>(input)->getOutput(0);
  auto result = createOp<ops::DivOp>(one, sqrt)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertSqrt(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::SqrtOp>(input)->getOutput(0);
  return {result};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertLogistic(const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::SigmoidOp>(input);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertTranspose(const tflite::TransposeOptionsT * /*opts*/,
                                  const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);
  mir::Tensor<int32_t> perm_tensor(extractTensor(inputs.at(1)));

  std::vector<std::size_t> axis_order = convertIntTensorToVector<std::size_t>(perm_tensor);
  auto result = createOp<ops::TransposeOp>(input, axis_order);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertStridedSlice(const tflite::StridedSliceOptionsT *opts,
                                     const std::vector<mir::Operation::Output *> &inputs)
{
  if (opts->ellipsis_mask != 0)
    throw std::runtime_error("StridedSlice: parameter 'ellipsis_mask' is not supported.");

  if (opts->new_axis_mask != 0)
    throw std::runtime_error("StridedSlice: parameter 'new_axis_mask' is not supported.");

  auto input = inputs.at(0);
  mir::Tensor<int32_t> begin_tensor(extractTensor(inputs.at(1)));
  mir::Tensor<int32_t> end_tensor(extractTensor(inputs.at(2)));
  mir::Tensor<int32_t> strides_tensor(extractTensor(inputs.at(3)));

  std::vector<int32_t> begin = convertIntTensorToVector<int32_t>(begin_tensor);
  std::vector<int32_t> end = convertIntTensorToVector<int32_t>(end_tensor);
  std::vector<int32_t> strides = convertIntTensorToVector<int32_t>(strides_tensor);

  int32_t begin_mask = opts->begin_mask;
  int32_t end_mask = opts->end_mask;
  int32_t shrink_axis_mask = opts->shrink_axis_mask;

  const auto &input_shape = input->getShape();
  int32_t num_dims = input_shape.rank();

  for (int32_t stride : strides)
  {
    if (stride != 1)
      throw std::runtime_error("StridedSlice: parameter 'strides' is not supported");
  }

  Shape start(num_dims);
  Shape size(num_dims);
  std::vector<int32_t> squeeze_dims;
  for (int axis = 0; axis < num_dims; axis++)
  {
    if (static_cast<uint32_t>(begin_mask) & (1u << static_cast<uint32_t>(axis)))
      start.dim(axis) = 0;
    else
      start.dim(axis) = begin.at(static_cast<uint64_t>(axis));

    if (static_cast<uint32_t>(end_mask) & (1u << static_cast<uint32_t>(axis)))
      size.dim(axis) = input_shape.dim(axis) - start.dim(axis);
    else
      size.dim(axis) = end.at(static_cast<uint64_t>(axis)) - start.dim(axis);

    if (static_cast<uint32_t>(shrink_axis_mask) & (1u << static_cast<uint32_t>(axis)))
      squeeze_dims.push_back(axis);
  }

  auto result = createOp<ops::SliceOp>(input, start, size);
  result = createOp<ops::SqueezeOp>(result->getOutput(0), squeeze_dims);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertLeakyReLU(const tflite::LeakyReluOptionsT *opts,
                                  const std::vector<mir::Operation::Output *> &inputs)
{
  auto input = inputs.at(0);

  auto result = createOp<ops::LeakyReluOp>(input, opts->alpha);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertShape(const tflite::ShapeOptionsT *opts,
                              const std::vector<mir::Operation::Output *> &inputs)
{
  if (opts->out_type != tflite::TensorType_INT32)
  {
    throw std::runtime_error(std::string("SHAPE: Unsupported tensor type: ") +
                             EnumNameTensorType(opts->out_type));
  }

  const auto &input_shape = inputs[0]->getShape();
  int32_t rank = input_shape.rank();
  std::vector<int32_t> data;
  data.reserve(static_cast<uint64_t>(rank));
  for (int32_t i = 0; i < rank; i++)
    data.emplace_back(input_shape.dim(i));
  mir::TensorVariant tensor({mir::DataType::INT32, {rank}}, data.data());
  auto result = createOp<ops::ConstantOp>(tensor);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
TFLiteOpCreator::convertHardSwish(const tflite::HardSwishOptionsT *,
                                  const std::vector<mir::Operation::Output *> &inputs)
{
  auto result = createOp<ops::HardSwishOp>(inputs[0])->getOutput(0);
  return {result};
}

} // namespace mir_tflite
