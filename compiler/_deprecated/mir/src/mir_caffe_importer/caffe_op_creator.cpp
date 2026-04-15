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

#include "caffe_op_creator.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/EluOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/GatherOp.h"
#include "mir/ops/LeakyReluOp.h"
#include "mir/ops/MaxOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/SigmoidOp.h"
#include "mir/ops/SliceOp.h"
#include "mir/ops/SoftmaxOp.h"
#include "mir/ops/TanhOp.h"
#include "mir/ops/TransposeOp.h"
#include "mir/Index.h"
#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

namespace mir_caffe
{

static mir::Shape convertBlobShape(const caffe::BlobShape &shape)
{
  mir::Shape mir_shape(shape.dim_size());

  for (int i = 0; i < shape.dim_size(); ++i)
  {
    mir_shape.dim(i) = shape.dim(i);
  }

  return mir_shape;
}

using namespace mir;

/// @brief Split arg into @p num_parts equal parts along @p axis axis.
std::vector<mir::Operation::Output *> CaffeOpCreator::createSplit(mir::Operation::Output *arg,
                                                                  int32_t num_parts, int32_t axis)
{
  const auto &arg_shape = arg->getShape();

  assert(axis >= 0 && axis < arg_shape.rank());
  int32_t part_size = arg_shape.dim(axis) / num_parts;
  assert(part_size * num_parts == arg_shape.dim(axis));

  Shape starts(arg_shape.rank());
  Shape sizes(arg_shape);
  sizes.dim(axis) = part_size;

  std::vector<mir::Operation::Output *> outputs(num_parts);
  for (int32_t i = 0; i < num_parts; ++i)
  {
    outputs[i] = createOp<ops::SliceOp>(arg, starts, sizes)->getOutput(0);
    starts.dim(axis) += part_size;
  }

  return outputs;
}

/// @brief Helper function for creating FullyConnected operation with non-square input.
mir::Operation::Output *CaffeOpCreator::createFullyConnected(mir::Operation::Output *input,
                                                             mir::Operation::Output *weights,
                                                             int32_t axis)
{
  const auto &input_shape = input->getShape();
  const auto &weights_shape = weights->getShape();

  assert(axis >= 0 && axis < input_shape.rank());
  assert(weights_shape.rank() == 2);

  // Result shape is: input.shape[0:axis] + weights.shape[1].
  Shape result_shape = input_shape;
  result_shape.resize(axis + 1);
  result_shape.dim(axis) = weights_shape.dim(1);

  // Flatten input to 2-D shape.
  int32_t outer_size = 1;
  for (int32_t i = 0; i < axis; ++i)
    outer_size *= input_shape.dim(i);
  int32_t inner_size = 1;
  for (int32_t i = axis; i < input_shape.rank(); ++i)
    inner_size *= input_shape.dim(i);

  auto flatten = createOp<ops::ReshapeOp>(input, Shape{outer_size, inner_size})->getOutput(0);
  auto fc = createOp<ops::FullyConnectedOp>(flatten, weights)->getOutput(0);
  return createOp<ops::ReshapeOp>(fc, result_shape)->getOutput(0);
}

TensorVariant CaffeOpCreator::convertBlob(const caffe::BlobProto &blob)
{
  const void *src_data;

  mir::DataType dtype;
  if (blob.data_size() != 0)
  {
    assert(blob.double_data_size() == 0);
    dtype = mir::DataType::FLOAT32;
    src_data = blob.data().data();
  }
  else if (blob.double_data_size() != 0)
  {
    dtype = mir::DataType::FLOAT64;
    src_data = blob.double_data().data();
  }
  else
  {
    throw std::runtime_error("No data in Caffe BlobProto, investigate");
  }

  const mir::Shape shape = convertBlobShape(blob.shape());
  return TensorVariant({dtype, shape}, src_data);
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertInput(const caffe::LayerParameter &layer)
{
  const auto &params = layer.input_param();
  const auto num_inputs = layer.top_size();
  const auto num_shapes = params.shape_size();
  std::vector<mir::Operation::Output *> outputs;

  assert((num_shapes == 1 || num_shapes == num_inputs) && "Unsupported number of shapes.");

  for (int i = 0; i < num_inputs; ++i)
  {
    const auto &blob_shape = params.shape(num_shapes == 1 ? 0 : i);
    mir::TensorType input_type(DataType::FLOAT32, convertBlobShape(blob_shape));
    auto input = createOp<ops::InputOp>(input_type)->getOutput(0);
    outputs.push_back(input);
  }

  return outputs;
}

template <class OperationAttributes>
static void convertConvolutionParam(const caffe::ConvolutionParameter &conv_param,
                                    OperationAttributes &attributes)
{
  std::int32_t stride_h, stride_w;
  if (conv_param.has_stride_h() || conv_param.has_stride_w())
  {
    // If stride_h or stride_w are set, they take precedence.
    stride_h = conv_param.stride_h();
    stride_w = conv_param.stride_w();
  }
  else if (conv_param.stride_size() == 0)
  {
    // If no strides specified, they defaults to 1.
    stride_h = stride_w = 1;
  }
  else if (conv_param.stride_size() == 1)
  {
    // If only one stride specified, all strides take the same value.
    stride_h = stride_w = conv_param.stride(0);
  }
  else
  {
    // Otherwise, there must be a stride for each dimension.
    assert(conv_param.stride_size() == 2);
    stride_h = conv_param.stride(0);
    stride_w = conv_param.stride(1);
  }
  attributes.strides = {stride_h, stride_w};

  std::int32_t pad_h, pad_w;
  if (conv_param.has_pad_h() || conv_param.has_pad_w())
  {
    // If pad_h or pad_w are set, they take precedence.
    pad_h = conv_param.pad_h();
    pad_w = conv_param.pad_w();
  }
  else if (conv_param.pad_size() == 0)
  {
    // If no pads specified, they defaults to 0.
    pad_h = pad_w = 0;
  }
  else if (conv_param.pad_size() == 1)
  {
    // If only one pad specified, all pads take the same value.
    pad_h = pad_w = conv_param.pad(0);
  }
  else
  {
    // Otherwise, there must be a pad for each dimension.
    assert(conv_param.pad_size() == 2);
    pad_h = conv_param.pad(0);
    pad_w = conv_param.pad(1);
  }
  attributes.padding_after = attributes.padding_before = {pad_h, pad_w};
}

void CaffeOpCreator::checkConvolution(const caffe::LayerParameter &layer,
                                      std::set<std::string> &problems_ops_set)
{
  const caffe::ConvolutionParameter &params = layer.convolution_param();

  assert(params.stride_size() <= 2);

  if (params.axis() != 1)
    problems_ops_set.insert("Conv2D: Unsupported axis");

  if (params.pad_size() != 0 && (params.has_pad_h() || params.has_pad_w()))
    problems_ops_set.insert("Conv2D: Conflicting padding properties");

  if (params.pad_size() > 2)
    problems_ops_set.insert("Conv2D: Unsupported number of pads");
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertConvolution(const caffe::LayerParameter &layer,
                                   const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.convolution_param();
  Conv2DOpAttributes attributes;

  convertConvolutionParam(params, attributes);
  attributes.num_groups = params.group();
  attributes.data_format = DataFormat::NCHW;

  assert(layer.blobs(0).shape().dim_size() == 4);
  auto kernel = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)))->getOutput(0);
  std::vector<std::size_t> perm{0, 2, 3, 1}; // OIHW -> OHWI
  kernel = createOp<ops::TransposeOp>(kernel, perm)->getOutput(0);
  auto result = createOp<ops::Conv2DOp>(inputs[0], kernel, attributes)->getOutput(0);

  // Add the bias, if any.
  if (params.bias_term())
  {
    auto bias = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
    bias = createOp<ops::ReshapeOp>(bias, Shape{1, bias->getShape().dim(0), 1, 1})->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertDeconvolution(const caffe::LayerParameter &layer,
                                     const std::vector<mir::Operation::Output *> &inputs)
{
  const caffe::ConvolutionParameter &params = layer.convolution_param();
  Deconv2DOpAttributes attributes;

  convertConvolutionParam(params, attributes);
  attributes.data_format = DataFormat::NCHW;

  if (params.group() != 1)
  {
    throw std::runtime_error("Deconvolution: 'group' != 1 is not supported.");
  }

  auto kernel = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)))->getOutput(0);
  std::vector<std::size_t> perm{2, 3, 1, 0}; // IOHW -> HWOI
  kernel = createOp<ops::TransposeOp>(kernel, perm)->getOutput(0);
  auto result = createOp<ops::DeConv2DOp>(inputs[0], kernel, attributes)->getOutput(0);

  // bias_term is optional (so might not be present) and defaults to true
  if (params.bias_term())
  {
    auto bias = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
    bias = createOp<ops::ReshapeOp>(bias, Shape{1, bias->getShape().dim(0), 1, 1})->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertInnerProduct(const caffe::LayerParameter &layer,
                                    const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.inner_product_param();
  auto weights = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)))->getOutput(0);

  if (!params.transpose())
    weights = createOp<ops::TransposeOp>(weights, std::vector<std::size_t>{1, 0})->getOutput(0);

  auto result = createFullyConnected(inputs[0], weights, params.axis());

  // Add the bias, if any.
  if (params.bias_term())
  {
    auto bias = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertConcat(const caffe::LayerParameter &layer,
                              const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.concat_param();
  auto concat = createOp<ops::ConcatOp>(inputs, params.axis());
  return {concat->getOutput(0)};
}

template <class PoolingAttributes>
static void convertPoolingParam(const caffe::PoolingParameter &params,
                                const mir::Shape &input_shape, PoolingAttributes &attributes)
{
  std::int32_t kernel_h, kernel_w;
  assert(!params.global_pooling());
  if (params.has_kernel_size())
  {
    kernel_h = kernel_w = params.kernel_size();
  }
  else
  {
    kernel_h = params.kernel_h();
    kernel_w = params.kernel_w();
  }
  attributes.window = {kernel_h, kernel_w};

  std::int32_t stride_h, stride_w;
  if (params.has_stride_h() || params.has_stride_w())
  {
    stride_h = params.stride_h();
    stride_w = params.stride_w();
  }
  else
  {
    stride_h = stride_w = params.stride();
  }
  attributes.strides = {stride_h, stride_w};

  std::int32_t pad_h, pad_w;
  if (params.has_pad_h() || params.has_pad_w())
  {
    pad_h = params.pad_h();
    pad_w = params.pad_w();
  }
  else
  {
    pad_h = pad_w = params.pad();
  }

  attributes.padding_before = attributes.padding_after = {pad_h, pad_w};

  // Caffe uses different formula for computing output shape than MIR. Adjust padding so that
  // the output shape stays the same.
  constexpr int num_spatial_dims = 2;
  for (int i = 0; i < num_spatial_dims; ++i)
  {
    // Assuming NCHW format.
    const std::int32_t padded_input =
      input_shape.dim(2 + i) + attributes.padding_before[i] + attributes.padding_after[i];
    if ((padded_input - attributes.window[i]) % attributes.strides[i] != 0)
      ++attributes.padding_after[i];
  }
}

void CaffeOpCreator::checkPooling(const caffe::LayerParameter &layer,
                                  std::set<std::string> &problems_ops_set)
{
  const caffe::PoolingParameter &params = layer.pooling_param();

  if (params.has_global_pooling() && params.global_pooling())
    problems_ops_set.insert("Pooling: pooling layer global_pooling param is not supported yet");

  if (params.pool() != caffe::PoolingParameter::AVE &&
      params.pool() != caffe::PoolingParameter::MAX)
    problems_ops_set.insert("Pooling: unsupported pooling type");

  if (params.has_pad() && (params.has_pad_h() || params.has_pad_w()))
    problems_ops_set.insert("Pooling: conflicting padding properties in pooling");
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertPooling(const caffe::LayerParameter &layer,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.pooling_param();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  mir::Operation::Output *result;

  switch (params.pool())
  {
    case caffe::PoolingParameter::AVE:
    {
      AvgPool2DOpAttributes attributes_avg;
      attributes_avg.data_format = DataFormat::NCHW;
      convertPoolingParam(params, input->getShape(), attributes_avg);
      result = createOp<ops::AvgPool2DOp>(input, attributes_avg)->getOutput(0);
      break;
    }
    case caffe::PoolingParameter::MAX:
    {
      MaxPool2DOpAttributes attributes_max;
      attributes_max.data_format = DataFormat::NCHW;
      convertPoolingParam(params, input->getShape(), attributes_max);
      result = createOp<ops::MaxPool2DOp>(input, attributes_max)->getOutput(0);
      break;
    }
    default:
      throw std::runtime_error("Unsupported PoolMethod: " + std::to_string(params.pool()));
  }

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertSoftmax(const caffe::LayerParameter &layer,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.softmax_param();

  // CPP and ACL backends are able to perform Softmax only along the last axis.
  // FIXME Do it in backends.
  if (inputs[0]->getShape().rank() == 4)
  {
    // For now, we only account for the most common case.
    if (params.axis() != 1)
      throw std::runtime_error("Softmax: unsupported axis");
    int32_t axis = 3;
    auto input = createOp<ops::TransposeOp>(inputs[0], std::vector<std::size_t>{0, 2, 3, 1});
    auto softmax = createOp<ops::SoftmaxOp>(input->getOutput(0), axis);
    auto result =
      createOp<ops::TransposeOp>(softmax->getOutput(0), std::vector<std::size_t>{0, 3, 1, 2});
    return {result->getOutput(0)};
  }

  auto softmax = createOp<ops::SoftmaxOp>(inputs[0], params.axis());
  return {softmax->getOutput(0)};
}

void CaffeOpCreator::checkReshape(const caffe::LayerParameter &layer,
                                  std::set<std::string> &problems_ops_set)
{
  const caffe::ReshapeParameter &params = layer.reshape_param();

  if (params.has_axis() || params.has_num_axes())
    problems_ops_set.insert("Reshape layer axis and num_axes params are not supported yet");

  if (!params.has_shape())
    problems_ops_set.insert("Reshape layer doesn't have shape parameter");

  const mir::Shape newShape = convertBlobShape(params.shape());

  for (int32_t i = 0; i < newShape.rank(); ++i)
    if (newShape.dim(i) == 0)
      problems_ops_set.insert("Reshape layer zero shape values are not supported yet");
}

/**
 * @brief Converts Caffe Reshape layer to Model IR Reshape operation.
 * @todo Support "axis" and "num_axes" parameters as needed.
 * @todo Decide how to react to the absence of "shape" parameter.
 * @todo Support zero values in "shape" parameter.
 */
std::vector<mir::Operation::Output *>
CaffeOpCreator::convertReshape(const caffe::LayerParameter &layer,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  const caffe::ReshapeParameter &params = layer.reshape_param();

  const mir::Shape new_shape = convertBlobShape(params.shape());
  auto reshape = createOp<ops::ReshapeOp>(inputs[0], new_shape);
  return {reshape->getOutput(0)};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertReLU(const caffe::LayerParameter &layer,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  mir::Operation *relu;
  if (layer.relu_param().has_negative_slope())
  {
    float alpha = layer.relu_param().negative_slope();
    relu = createOp<ops::LeakyReluOp>(inputs[0], alpha);
  }
  else
  {
    relu = createOp<ops::ReluOp>(inputs[0]);
  }

  return {relu->getOutput(0)};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertScale(const caffe::LayerParameter &layer,
                             const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.scale_param();
  auto scale = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)))->getOutput(0);
  scale = createOp<ops::ReshapeOp>(scale, Shape{1, scale->getShape().dim(0), 1, 1})->getOutput(0);
  auto result = createOp<ops::MulOp>(inputs[0], scale)->getOutput(0);

  // Add the bias, if any.
  if (params.bias_term())
  {
    auto bias = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
    bias = createOp<ops::ReshapeOp>(bias, Shape{1, bias->getShape().dim(0), 1, 1})->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

void CaffeOpCreator::checkBatchNorm(const caffe::LayerParameter &layer,
                                    std::set<std::string> &problems_ops_set)
{
  const auto &scale_shape = layer.blobs(2).shape();

  // Check that last blob(with scaleFactor) containing only one number
  if (scale_shape.dim_size() != 1 || scale_shape.dim(0) != 1)
    problems_ops_set.insert("Unexpected shape of scale parameter in batch norm");
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertBatchNorm(const caffe::LayerParameter &layer,
                                 const std::vector<mir::Operation::Output *> &inputs)
{
  const caffe::BatchNormParameter &params = layer.batch_norm_param();

  auto input = inputs[0];
  auto mean_tensor = convertBlob(layer.blobs(0));
  auto var_tensor = convertBlob(layer.blobs(1));
  auto scale_tensor = convertBlob(layer.blobs(2));
  const float eps = params.eps();

  float scale_factor = *reinterpret_cast<float *>(scale_tensor.at(mir::Index{0}));

  // See https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cpp#L100
  // Y = (X - mean / scale_factor) / sqrt(var / scale_factor + epsilon) =
  //   = (X + C1) * C2
  if (scale_factor != 0.0f)
    scale_factor = 1.0f / scale_factor;

  // C1 = -mean / scale_factor
  Tensor<float> mean_accessor(mean_tensor);
  for (const auto &idx : ShapeRange(mean_accessor.getShape()))
    mean_accessor.at(idx) *= -scale_factor;
  auto c1 = createOp<ops::ConstantOp>(mean_tensor)->getOutput(0);

  // C2 = 1 / sqrt(var / scale_factor + epsilon)
  Tensor<float> var_accessor(var_tensor);
  for (const auto &idx : ShapeRange(var_accessor.getShape()))
    var_accessor.at(idx) = 1.0f / std::sqrt(var_accessor.at(idx) * scale_factor + eps);
  auto c2 = createOp<ops::ConstantOp>(var_tensor)->getOutput(0);

  c1 = createOp<ops::ReshapeOp>(c1, Shape{1, c1->getShape().dim(0), 1, 1})->getOutput(0);
  c2 = createOp<ops::ReshapeOp>(c2, Shape{1, c2->getShape().dim(0), 1, 1})->getOutput(0);

  // Y = (X + C1) * C2
  auto result = createOp<ops::AddOp>(input, c1)->getOutput(0);
  result = createOp<ops::MulOp>(result, c2)->getOutput(0);

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertDropout(const caffe::LayerParameter &,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  // This is a no-op in inference mode.
  return {inputs[0]};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertELU(const caffe::LayerParameter &layer,
                           const std::vector<mir::Operation::Output *> &inputs)
{
  const caffe::ELUParameter &params = layer.elu_param();

  auto elu = createOp<ops::EluOp>(inputs[0], params.alpha());
  return {elu->getOutput(0)};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertEmbed(const caffe::LayerParameter &layer,
                             const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.embed_param();
  auto data = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)));
  auto result = createOp<ops::GatherOp>(data->getOutput(0), inputs[0], 0)->getOutput(0);

  // Add the bias, if any.
  if (params.bias_term())
  {
    auto bias = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
    result = createOp<ops::AddOp>(result, bias)->getOutput(0);
  }

  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertSigmoid(const caffe::LayerParameter &,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  auto result = createOp<ops::SigmoidOp>(inputs[0]);
  return {result->getOutput(0)};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertTanH(const caffe::LayerParameter &,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  auto tanh = createOp<ops::TanhOp>(inputs[0]);
  return {tanh->getOutput(0)};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertEltwise(const caffe::LayerParameter &layer,
                               const std::vector<mir::Operation::Output *> &inputs)
{
  auto &params = layer.eltwise_param();

  mir::Operation::Output *result;
  switch (params.operation())
  {
    case caffe::EltwiseParameter::PROD:
    {
      result = createOp<ops::MulOp>(inputs[0], inputs[1])->getOutput(0);
      for (int i = 2; i < layer.bottom_size(); ++i)
      {
        result = createOp<ops::MulOp>(result, inputs[i])->getOutput(0);
      }
      break;
    }
    case caffe::EltwiseParameter::SUM:
    {
      std::vector<mir::Operation::Output *> scaled_inputs = inputs;
      if (params.coeff_size() > 0)
      {
        assert(params.coeff_size() == layer.bottom_size());
        for (int i = 0; i < layer.bottom_size(); i++)
        {
          if (params.coeff(i) != 1.0f)
          {
            const float coeff_val = params.coeff(i);
            TensorVariant coeff_tensor({DataType::FLOAT32, {}}, &coeff_val);
            auto coeff_const = createOp<ops::ConstantOp>(coeff_tensor)->getOutput(0);
            scaled_inputs[i] = createOp<ops::MulOp>(coeff_const, inputs[i])->getOutput(0);
          }
        }
      }
      result = createOp<ops::AddOp>(scaled_inputs[0], scaled_inputs[1])->getOutput(0);
      for (int i = 2; i < layer.bottom_size(); ++i)
      {
        result = createOp<ops::AddOp>(result, scaled_inputs[i])->getOutput(0);
      }
      break;
    }
    case caffe::EltwiseParameter::MAX:
    {
      result = createOp<ops::MaxOp>(inputs[0], inputs[1])->getOutput(0);
      for (int i = 2; i < layer.bottom_size(); ++i)
      {
        result = createOp<ops::MaxOp>(result, inputs[i])->getOutput(0);
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown element-wise operation.");
  }
  return {result};
}

std::vector<mir::Operation::Output *>
CaffeOpCreator::convertSplit(const caffe::LayerParameter &layer,
                             const std::vector<mir::Operation::Output *> &inputs)
{
  std::vector<mir::Operation::Output *> outputs(layer.top_size(), inputs.at(0));
  return outputs;
}

void CaffeOpCreator::checkLSTM(const caffe::LayerParameter &layer,
                               std::set<std::string> &problems_ops_set)
{
  const auto &params = layer.recurrent_param();
  if (params.expose_hidden())
    problems_ops_set.insert("LSTM: parameter 'expose_hidden' has unsupported value: " +
                            std::to_string(params.expose_hidden()));
}

static TensorVariant createZeroedTensor(const mir::Shape &shape)
{
  // TODO For now it is hardcoded float32.
  auto elem_type = mir::DataType::FLOAT32;
  std::vector<float> zeros(static_cast<std::size_t>(shape.numElements()), 0.0f);
  return TensorVariant({elem_type, shape}, zeros.data());
}

/* See the following links for details on implementation:
 * https://github.com/BVLC/caffe/blob/master/src/caffe/layers/recurrent_layer.cpp
 * https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_layer.cpp
 * https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_unit_layer.cpp
 *
 * Inputs:
 *   x        -- The time-varying input. Shape: [T, N, d0, d1, ..., dn].
 *   cont     -- The sequence continuation indicators. Shape: [T, N].
 *   x_static -- The static (non-time-varying) input. Shape: [N, ...].
 *               This parameter is optional and not currently supported.
 *
 * Additional inputs when parameter "expose_hidden" is true (not currently supported):
 *   h_0  -- The initial value of the hidden state. Shape: [1, N, D].
 *   c_0  -- The initial value of the cell state. Shape: [1, N, D].
 *
 * Learned parameters:
 *   xw -- x weights for input, output, forget and cell gates concatenated.
 *         Shape: [4 * D, d0 * d1 * ... * dn].
 *   xb -- x biases for input, output, forget and cell gates concatenated. Shape: [4 * D].
 *   hw -- h weights for input, output, forget and cell gates concatenated. Shape: [4 * D, D].
 *
 * Outputs:
 *   h   -- The time-varying output. Shape: [T, N, D].
 *
 * Additional outputs when parameter "expose_hidden" is true (not currently supported):
 *   h_T -- The value of the hidden state at the last timestep. Shape: [1, N, D].
 *   c_T -- The value of the cell state at the last timestep. Shape: [1, N, D].
 *
 * Here:
 *   T - the number of timesteps,
 *   N - the number of independent streams.
 *   D - the number of hidden parameters.
 *
 * Formulas:
 *   c_cont = c[t-1] * cont[t]
 *   h_cont = h[t-1] * cont[t]
 *   i[t] = Sigmoid(x[t] . xw_i + xb_i + h_cont . hw_i)
 *   f[t] = Sigmoid(x[t] . xw_f + xb_f + h_cont . hw_f)
 *   o[t] = Sigmoid(x[t] . xw_o + xb_o + h_cont . hw_o)
 *   g[t] =    Tanh(x[t] . xw_g + xb_g + h_cont . hw_g)
 *   c[t] = c_cont * f[t] + i[t] * g[t]
 *   h[t] = o[t] * Tanh(c[t])
 *
 * Here:
 *   t -- the timestep (ranges from 1 to T),
 *   * -- the inner product,
 *   . -- the Hadamard product (elementwise product).
 *
 * In this implementation the inner products for all gates are performed as single inner product for
 * efficiency.
 */
std::vector<mir::Operation::Output *>
CaffeOpCreator::convertLSTM(const caffe::LayerParameter &layer,
                            const std::vector<mir::Operation::Output *> &inputs)
{
  const auto &params = layer.recurrent_param();

  // Inputs to the layer.
  auto x = inputs[0];
  auto cont = inputs[1];
  assert(inputs.size() == 2);

  const auto &x_shape = x->getShape();
  const int32_t seq_length = x_shape.dim(0);
  const int32_t batch_size = x_shape.dim(1);
  const int32_t hidden_size = params.num_output();

  // Learned parameters of the layer. Tensors are transposed to match the ModelIR.
  auto xw = createOp<ops::ConstantOp>(convertBlob(layer.blobs(0)))->getOutput(0);
  auto xb = createOp<ops::ConstantOp>(convertBlob(layer.blobs(1)))->getOutput(0);
  auto hw = createOp<ops::ConstantOp>(convertBlob(layer.blobs(2)))->getOutput(0);
  xw = createOp<ops::TransposeOp>(xw, std::vector<std::size_t>{1, 0})->getOutput(0);
  hw = createOp<ops::TransposeOp>(hw, std::vector<std::size_t>{1, 0})->getOutput(0);

  // Add a dummy dimension so that element-wise operations perform properly.
  cont = createOp<ops::ReshapeOp>(cont, Shape{seq_length, batch_size, 1})->getOutput(0);

  // Initialize cell and hidden states with zeros.
  auto zero_tensor = createZeroedTensor(Shape{1, batch_size, hidden_size});
  auto c_t = createOp<ops::ConstantOp>(zero_tensor)->getOutput(0);
  auto h_t = createOp<ops::ConstantOp>(zero_tensor)->getOutput(0);

  auto x_xw = createFullyConnected(x, xw, 2);
  auto x_xw_b = createOp<ops::AddOp>(x_xw, xb)->getOutput(0);

  // Split input and continuation tensors into seq_length slices.
  std::vector<mir::Operation::Output *> x_xw_b_slices = createSplit(x_xw_b, seq_length, 0);
  std::vector<mir::Operation::Output *> cont_slices = createSplit(cont, seq_length, 0);
  std::vector<mir::Operation::Output *> h_slices(seq_length);

  for (int32_t t = 0; t < seq_length; t++)
  {
    auto c_cont_t = createOp<ops::MulOp>(c_t, cont_slices[t])->getOutput(0);
    auto h_cont_t = createOp<ops::MulOp>(h_t, cont_slices[t])->getOutput(0);

    auto x_xw_b_t = x_xw_b_slices[t];
    auto h_hw_t = createFullyConnected(h_cont_t, hw, 2);
    auto activation_inputs_concat = createOp<ops::AddOp>(x_xw_b_t, h_hw_t)->getOutput(0);
    auto activation_inputs = createSplit(activation_inputs_concat, 4, 2);

    auto i_t = createOp<ops::SigmoidOp>(activation_inputs[0])->getOutput(0);
    auto f_t = createOp<ops::SigmoidOp>(activation_inputs[1])->getOutput(0);
    auto o_t = createOp<ops::SigmoidOp>(activation_inputs[2])->getOutput(0);
    auto g_t = createOp<ops::TanhOp>(activation_inputs[3])->getOutput(0);

    c_t = createOp<ops::AddOp>(createOp<ops::MulOp>(c_cont_t, f_t)->getOutput(0),
                               createOp<ops::MulOp>(i_t, g_t)->getOutput(0))
            ->getOutput(0);
    h_t = createOp<ops::MulOp>(createOp<ops::TanhOp>(c_t)->getOutput(0), o_t)->getOutput(0);

    h_slices[t] = h_t;
  }

  return {createOp<ops::ConcatOp>(h_slices, 0)->getOutput(0)};
}

} // namespace mir_caffe
