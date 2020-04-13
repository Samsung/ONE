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

#include "mir2loco.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/Deconv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/DivOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/SoftmaxOp.h"
#include "mir/ops/SubOp.h"
#include "mir/ops/TransposeOp.h"

#include "mir/ShapeRange.h"

#include <cassert>
#include <cstring>
#include <stdex/Memory.h>

namespace mir2loco
{
namespace
{
template <class NodeType> void setupShape(const mir::Shape &shape, NodeType *node)
{
  node->rank(shape.rank());
  for (int32_t i = 0; i < shape.rank(); i++)
  {
    node->dim(i) = static_cast<uint32_t>(shape.dim(i));
  }
}

std::unique_ptr<loco::TensorShape> make_tensor_shape(const mir::Shape &shape)
{
  auto res = stdex::make_unique<loco::TensorShape>();
  setupShape(shape, res.get());
  return std::move(res);
}

void setupPad(const std::vector<std::int32_t> &padding_before,
              const std::vector<std::int32_t> &padding_after, loco::Padding2D *pad)
{
  assert(padding_before.size() == 2 && padding_after.size() == 2);
  pad->top(padding_before[0]);
  pad->left(padding_before[1]);
  pad->bottom(padding_after[0]);
  pad->right(padding_after[1]);
}

void setupWindow(const std::vector<std::int32_t> &window_size, loco::Window<2> *window)
{
  assert(window_size.size() == 2);
  window->vertical(window_size[0]);
  window->horizontal(window_size[1]);
}

void setupStride(const std::vector<std::int32_t> &strides, loco::Stride<2> *stride)
{
  assert(strides.size() == 2);
  stride->vertical(strides[0]);
  stride->horizontal(strides[1]);
}

loco::Permutation<loco::Domain::Feature> createFeaturePermutation(mir::DataFormat format)
{
  loco::Permutation<loco::Domain::Feature> perm;
  if (format == mir::DataFormat::NHWC)
  {
    perm.axis(loco::FeatureAxis::Count) = 0;
    perm.axis(loco::FeatureAxis::Height) = 1;
    perm.axis(loco::FeatureAxis::Width) = 2;
    perm.axis(loco::FeatureAxis::Depth) = 3;
  }
  else
  {
    assert(format == mir::DataFormat::NCHW);
    perm.axis(loco::FeatureAxis::Count) = 0;
    perm.axis(loco::FeatureAxis::Depth) = 1;
    perm.axis(loco::FeatureAxis::Height) = 2;
    perm.axis(loco::FeatureAxis::Width) = 3;
  }
  return perm;
}

std::unique_ptr<loco::FeatureEncoder> createFeatureEncoder(mir::DataFormat data_format)
{
  auto perm = createFeaturePermutation(data_format);
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::Feature>>(perm);
}

std::unique_ptr<loco::FeatureDecoder> createFeatureDecoder(mir::DataFormat data_format)
{
  auto perm = createFeaturePermutation(data_format);
  return stdex::make_unique<loco::PermutingDecoder<loco::Domain::Feature>>(perm);
}

std::unique_ptr<loco::FilterEncoder> createOHWIFilterEncoder()
{
  loco::Permutation<loco::Domain::Filter> perm;
  perm.axis(loco::FilterAxis::Count) = 0;
  perm.axis(loco::FilterAxis::Height) = 1;
  perm.axis(loco::FilterAxis::Width) = 2;
  perm.axis(loco::FilterAxis::Depth) = 3;
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>(perm);
}

std::unique_ptr<loco::FilterEncoder> createHWOIFilterEncoder()
{
  loco::Permutation<loco::Domain::Filter> perm;
  perm.axis(loco::FilterAxis::Height) = 0;
  perm.axis(loco::FilterAxis::Width) = 1;
  perm.axis(loco::FilterAxis::Count) = 2;
  perm.axis(loco::FilterAxis::Depth) = 3;
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>(perm);
}

std::unique_ptr<loco::DepthwiseFilterEncoder> createHWIMDepthwiseFilterEncoder()
{
  loco::Permutation<loco::Domain::DepthwiseFilter> perm;
  perm.axis(loco::DepthwiseFilterAxis::Height) = 0;
  perm.axis(loco::DepthwiseFilterAxis::Width) = 1;
  perm.axis(loco::DepthwiseFilterAxis::Depth) = 2;
  perm.axis(loco::DepthwiseFilterAxis::Multiplier) = 3;
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::DepthwiseFilter>>(perm);
}

std::unique_ptr<loco::DepthwiseFilterEncoder> createIHWMDepthwiseFilterEncoder()
{
  loco::Permutation<loco::Domain::DepthwiseFilter> perm;
  perm.axis(loco::DepthwiseFilterAxis::Depth) = 0;
  perm.axis(loco::DepthwiseFilterAxis::Height) = 1;
  perm.axis(loco::DepthwiseFilterAxis::Width) = 2;
  perm.axis(loco::DepthwiseFilterAxis::Multiplier) = 3;
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::DepthwiseFilter>>(perm);
}

std::unique_ptr<loco::MatrixEncoder> createHWMatrixEncoder()
{
  loco::Permutation<loco::Domain::Matrix> perm;
  perm.axis(loco::MatrixAxis::Height) = 0;
  perm.axis(loco::MatrixAxis::Width) = 1;
  return stdex::make_unique<loco::PermutingEncoder<loco::Domain::Matrix>>(perm);
}

std::unique_ptr<loco::MatrixDecoder> createHWMatrixDecoder()
{
  loco::Permutation<loco::Domain::Matrix> perm;
  perm.axis(loco::MatrixAxis::Height) = 0;
  perm.axis(loco::MatrixAxis::Width) = 1;
  return stdex::make_unique<loco::PermutingDecoder<loco::Domain::Matrix>>(perm);
}

loco::DataType convertDataType(mir::DataType data_type)
{
  switch (data_type)
  {
    case mir::DataType::UNKNOWN:
      return loco::DataType::Unknown;
    case mir::DataType::FLOAT32:
      return loco::DataType::FLOAT32;
    case mir::DataType::FLOAT64:
      return loco::DataType::FLOAT64;
    case mir::DataType::INT32:
      return loco::DataType::S32;
    case mir::DataType::INT64:
      return loco::DataType::S64;
    default:
      break;
  }
  throw std::runtime_error("Unsupported data type");
}

loco::Node *createBroadcastIfNeeded(loco::Node *node, const mir::Shape &shape,
                                    const mir::Shape &out_shape)
{
  auto graph = node->graph();

  if (shape == out_shape)
    return node; // not needed

  int32_t out_rank = out_shape.rank();
  int32_t rank_diff = out_rank - shape.rank();
  // Create Broadcast
  auto *broadcast = graph->nodes()->create<loco::TensorBroadcast>();
  // Create Reshape for equal ranks
  if (shape.rank() != out_rank)
  {
    auto *reshape = graph->nodes()->create<loco::FixedReshape>();
    reshape->input(node);
    reshape->rank(out_rank);
    broadcast->input(reshape);
    // Set reshape dims
    for (int32_t dim = 0; dim < out_rank; dim++)
    {
      if (dim < rank_diff)
        reshape->dim(dim) = 1;
      else
        reshape->dim(dim) = shape.dim(dim - rank_diff);
    }
  }
  else
  {
    broadcast->input(node);
  }
  // Flag if no one dim isn't equal
  bool compatible_shapes = true;
  for (int32_t dim = 0; dim < out_rank; dim++)
  {
    // Set broadcast mapping
    if (dim < rank_diff || (shape.dim(dim - rank_diff) == 1 && out_shape.dim(dim) != 1))
      broadcast->mapping()->dim(dim) = out_shape.dim(dim);
    // Check compatibility
    if (dim >= rank_diff && shape.dim(dim - rank_diff) != 1 &&
        shape.dim(dim - rank_diff) != out_shape.dim(dim))
      compatible_shapes = false;
  }
  // Check compatibility
  if (!compatible_shapes)
    throw std::runtime_error("Not compatible shapes for broadcasting!");

  return broadcast;
}

template <typename NodeType>
NodeType *createEltwiseBinary(const mir::ops::BinaryElementwiseOp &op, loco::Node *lhs,
                              loco::Node *rhs)
{
  auto graph = lhs->graph();

  const auto &lhs_shape = op.getInput(0)->getShape();
  const auto &rhs_shape = op.getInput(1)->getShape();
  const auto &out_shape = op.getOutputShape(0);
  // Create Broadcast if it's needed
  auto lhs_node = createBroadcastIfNeeded(lhs, lhs_shape, out_shape);
  auto rhs_node = createBroadcastIfNeeded(rhs, rhs_shape, out_shape);
  // Create Node
  auto result = graph->nodes()->create<NodeType>();
  result->lhs(lhs_node);
  result->rhs(rhs_node);
  return result;
}
} // namespace

void Transformer::visit(mir::ops::AddOp &op)
{
  // Get Input
  auto lhs = _mir2loco_map.at(op.getInput(0));
  auto rhs = _mir2loco_map.at(op.getInput(1));
  auto result = createEltwiseBinary<loco::EltwiseAdd>(op, lhs, rhs);
  // Not set Shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), result);
}

void Transformer::visit(mir::ops::AvgPool2DOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));

  auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
  encoded_input->input(input);
  encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

  auto *avg_pool_node = _loco_graph->nodes()->create<loco::AvgPool2D>();
  avg_pool_node->ifm(encoded_input);
  avg_pool_node->convention(op.getIncludePad() ? loco::AvgPool2D::Convention::Full
                                               : loco::AvgPool2D::Convention::Valid);
  setupWindow(op.getWindowSize(), avg_pool_node->window());
  setupStride(op.getStrides(), avg_pool_node->stride());
  setupPad(op.getPaddingBefore(), op.getPaddingAfter(), avg_pool_node->pad());

  auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
  output->input(avg_pool_node);
  output->decoder(createFeatureDecoder(op.getDataFormat()));

  _mir2loco_map.emplace(op.getOutput(0), output);
}

void Transformer::visit(mir::ops::ConcatOp &op)
{
  if (op.getNumInputs() < 2)
    throw std::runtime_error("Not enough tensors for concatenation!");

  loco::Node *last_concat = nullptr;

  for (std::size_t i = 1; i < op.getNumInputs(); i++)
  {
    loco::Node *lhs = last_concat;
    if (lhs == nullptr)
    {
      mir::Operation::Output *mir_lhs = op.getInput(i - 1);
      lhs = _mir2loco_map.at(mir_lhs);
    }
    mir::Operation::Output *mir_rhs = op.getInput(i);
    loco::Node *rhs = _mir2loco_map.at(mir_rhs);
    // Create TensorConcat
    auto concat_node = _loco_graph->nodes()->create<loco::TensorConcat>();
    // Set inputs
    concat_node->lhs(lhs);
    concat_node->rhs(rhs);
    // Set axis
    concat_node->axis(op.getAxis());
    // Set last concat
    last_concat = concat_node;
  }
  // Not set Shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), last_concat);
}

void Transformer::visit(mir::ops::ConstantOp &op)
{
  auto const_node = _loco_graph->nodes()->create<loco::ConstGen>();
  // Not set Input
  // Set Shape
  const auto &out_shape = op.getOutputShape(0);
  setupShape(out_shape, const_node);
  // Copy value
  const auto &value = op.getValue();
  const_node->dtype(convertDataType(value.getElementType()));
  // TODO Support other data types
  switch (const_node->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      const_node->size<loco::DataType::FLOAT32>(out_shape.numElements());
      float &const_float = const_node->at<loco::DataType::FLOAT32>(0);
      char *loco_ptr = reinterpret_cast<char *>(&const_float);
      char *mir_ptr = value.at(mir::Index(out_shape.rank()));
      std::memcpy(loco_ptr, mir_ptr, out_shape.numElements() * sizeof(float));
      break;
    }
    case loco::DataType::FLOAT64:
    {
      // TODO Change that when loco support other DataTypeImpl
      const_node->dtype(loco::DataType::FLOAT32);
      const_node->size<loco::DataType::FLOAT32>(out_shape.numElements());
      float &const_float = const_node->at<loco::DataType::FLOAT32>(0);
      char *mir_ptr = value.at(mir::Index(out_shape.rank()));
      double *mir_double = reinterpret_cast<double *>(mir_ptr);
      float *loco_float = &const_float;
      for (const mir::Index &idx : mir::ShapeRange(out_shape))
      {
        *loco_float = static_cast<float>(*mir_double);
        loco_float++;
        mir_double++;
      }
      break;
    }
    case loco::DataType::S32:
    {
      const_node->size<loco::DataType::S32>(out_shape.numElements());
      int32_t &const_int32 = const_node->at<loco::DataType::S32>(0);
      char *loco_ptr = reinterpret_cast<char *>(&const_int32);
      char *mir_ptr = value.at(mir::Index(out_shape.rank()));
      std::memcpy(loco_ptr, mir_ptr, out_shape.numElements() * sizeof(int32_t));
      break;
    }
    case loco::DataType::S64:
    {
      // TODO Change that when loco support other DataTypeImpl
      const_node->dtype(loco::DataType::S32);
      const_node->size<loco::DataType::S32>(out_shape.numElements());
      int32_t &const_int32 = const_node->at<loco::DataType::S32>(0);
      char *mir_ptr = value.at(mir::Index(out_shape.rank()));
      int64_t *mir_int64 = reinterpret_cast<int64_t *>(mir_ptr);
      int32_t *loco_int32 = &const_int32;
      for (const mir::Index &idx : mir::ShapeRange(out_shape))
      {
        *loco_int32 = static_cast<float>(*mir_int64);
        loco_int32++;
        mir_int64++;
      }
      break;
    }
    default:
      std::runtime_error("Unsupported data type");
  }
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), const_node);
}

void Transformer::visit(mir::ops::Conv2DOp &op)
{
  mir::Operation::Output *mir_input = op.getInput(0);
  mir::Operation::Output *mir_filter = op.getInput(1);

  loco::Node *input = _mir2loco_map.at(mir_input);
  loco::Node *filter = _mir2loco_map.at(mir_filter);

  // loco does not have grouped Conv2D operation. Try to translate into something else.
  if (op.getNumGroups() != 1)
  {
    const std::int32_t group_size = mir_filter->getShape().dim(3);
    const std::int32_t num_in_channels = group_size * op.getNumGroups();
    const std::int32_t num_out_channels = mir_filter->getShape().dim(0);

    // If the size of the group is 1, translate the operation into DepthwiseConv2D. Limit ourselves
    // with the case of 'multiplier' == 1 for now.
    if (group_size == 1 && (num_out_channels == num_in_channels))
    {
      // [O, H, W, I / group] == [I, H, W, M].
      auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
      encoded_input->input(input);
      encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

      auto *encoded_filter = _loco_graph->nodes()->create<loco::DepthwiseFilterEncode>();
      encoded_filter->input(filter);
      encoded_filter->encoder(createIHWMDepthwiseFilterEncoder());

      auto *dw_conv2d_node = _loco_graph->nodes()->create<loco::DepthwiseConv2D>();
      dw_conv2d_node->ifm(encoded_input);
      dw_conv2d_node->ker(encoded_filter);
      setupStride(op.getStrides(), dw_conv2d_node->stride());
      setupPad(op.getPaddingBefore(), op.getPaddingAfter(), dw_conv2d_node->pad());

      auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
      output->input(dw_conv2d_node);
      output->decoder(createFeatureDecoder(op.getDataFormat()));

      _mir2loco_map.emplace(op.getOutput(0), output);
    }
    else
    {
      // There are few things we can do here:
      // 1) If group_size == 1, reshape the kernel [O, H, W, I / group] == [I * M, H, W, 1] ->
      //    [I, M, H, W] and use DepthwiseConv2D.
      // 2) Split the operation into smaller Conv2Ds.
      // 3) Replicate the filter along 'O' axis 'num_groups' times, zero out some elements, and use
      //    ordinary Conv2D.
      throw std::runtime_error("Grouped Conv2D operation is not fully supported.");
    }
  }
  else
  {
    auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
    encoded_input->input(input);
    encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

    auto *encoded_filter = _loco_graph->nodes()->create<loco::FilterEncode>();
    encoded_filter->input(filter);
    encoded_filter->encoder(createOHWIFilterEncoder());

    auto *conv2d_node = _loco_graph->nodes()->create<loco::Conv2D>();
    conv2d_node->ifm(encoded_input);
    conv2d_node->ker(encoded_filter);
    setupStride(op.getStrides(), conv2d_node->stride());
    setupPad(op.getPaddingBefore(), op.getPaddingAfter(), conv2d_node->pad());

    auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
    output->input(conv2d_node);
    output->decoder(createFeatureDecoder(op.getDataFormat()));

    _mir2loco_map.emplace(op.getOutput(0), output);
  }
}

void Transformer::visit(mir::ops::DeConv2DOp &op)
{
  mir::Operation::Output *mir_input = op.getInput(0);
  mir::Operation::Output *mir_filter = op.getInput(1);

  loco::Node *input = _mir2loco_map.at(mir_input);
  loco::Node *filter = _mir2loco_map.at(mir_filter);

  auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
  encoded_input->input(input);
  encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

  auto *encoded_filter = _loco_graph->nodes()->create<loco::FilterEncode>();
  encoded_filter->input(filter);
  encoded_filter->encoder(createHWOIFilterEncoder());

  auto *tr_conv2d_node = _loco_graph->nodes()->create<loco::TransposedConv2D>();
  tr_conv2d_node->ifm(encoded_input);
  tr_conv2d_node->ker(encoded_filter);
  setupStride(op.getStrides(), tr_conv2d_node->stride());
  if (op.getPaddingType() == mir::ops::PaddingType::Explicit)
    setupPad(op.getPaddingBefore(), op.getPaddingAfter(), tr_conv2d_node->pad());
  else
    throw std::runtime_error("Not supported non explicit paddings on loco!");

  auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
  output->input(tr_conv2d_node);
  output->decoder(createFeatureDecoder(op.getDataFormat()));

  _mir2loco_map.emplace(op.getOutput(0), output);
}

void Transformer::visit(mir::ops::DepthwiseConv2DOp &op)
{
  mir::Operation::Output *mir_input = op.getInput(0);
  mir::Operation::Output *mir_filter = op.getInput(1);

  loco::Node *input = _mir2loco_map.at(mir_input);
  loco::Node *filter = _mir2loco_map.at(mir_filter);

  auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
  encoded_input->input(input);
  encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

  auto *encoded_filter = _loco_graph->nodes()->create<loco::DepthwiseFilterEncode>();
  encoded_filter->input(filter);
  encoded_filter->encoder(createHWIMDepthwiseFilterEncoder());

  auto *dw_conv2d_node = _loco_graph->nodes()->create<loco::DepthwiseConv2D>();
  dw_conv2d_node->ifm(encoded_input);
  dw_conv2d_node->ker(encoded_filter);
  setupStride(op.getStrides(), dw_conv2d_node->stride());
  setupPad(op.getPaddingBefore(), op.getPaddingAfter(), dw_conv2d_node->pad());

  auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
  output->input(dw_conv2d_node);
  output->decoder(createFeatureDecoder(op.getDataFormat()));

  _mir2loco_map.emplace(op.getOutput(0), output);
}

void Transformer::visit(mir::ops::DivOp &op)
{
  // Get Input
  loco::Node *lhs = _mir2loco_map.at(op.getInput(0));
  loco::Node *rhs = _mir2loco_map.at(op.getInput(1));
  auto result = createEltwiseBinary<loco::EltwiseDiv>(op, lhs, rhs);
  // Not set Shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), result);
}

void Transformer::visit(mir::ops::FullyConnectedOp &op)
{
  mir::Operation::Output *mir_lhs = op.getInput(0);
  mir::Operation::Output *mir_rhs = op.getInput(1);
  // Check 2D shape
  assert(op.getInput(0)->getShape().rank() == 2);
  assert(op.getInput(1)->getShape().rank() == 2);

  loco::Node *lhs = _mir2loco_map.at(mir_lhs);
  loco::Node *rhs = _mir2loco_map.at(mir_rhs);

  auto *encoded_lhs = _loco_graph->nodes()->create<loco::MatrixEncode>();
  encoded_lhs->input(lhs);
  encoded_lhs->encoder(createHWMatrixEncoder());

  auto *encoded_rhs = _loco_graph->nodes()->create<loco::MatrixEncode>();
  encoded_rhs->input(rhs);
  encoded_rhs->encoder(createHWMatrixEncoder());

  auto *mat_mul = _loco_graph->nodes()->create<loco::MatMul>();
  mat_mul->lhs(encoded_lhs);
  mat_mul->rhs(encoded_rhs);

  auto *output = _loco_graph->nodes()->create<loco::MatrixDecode>();
  output->input(mat_mul);
  output->decoder(createHWMatrixDecoder());

  _mir2loco_map.emplace(op.getOutput(0), output);
}

void Transformer::visit(mir::ops::InputOp &op)
{
  mir::Operation::Output *mir_output = op.getOutput(0);

  loco::GraphInput *graph_input = _loco_graph->inputs()->create();
  graph_input->name(mir_output->getName());
  graph_input->dtype(convertDataType(mir_output->getElementType()));

  auto *pull_node = _loco_graph->nodes()->create<loco::Pull>();
  setupShape(mir_output->getShape(), pull_node);

  loco::link(graph_input, pull_node);

  _mir2loco_map.emplace(mir_output, pull_node);
}

void Transformer::visit(mir::ops::MaxPool2DOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));

  auto *encoded_input = _loco_graph->nodes()->create<loco::FeatureEncode>();
  encoded_input->input(input);
  encoded_input->encoder(createFeatureEncoder(op.getDataFormat()));

  auto max_pool_node = _loco_graph->nodes()->create<loco::MaxPool2D>();
  max_pool_node->ifm(encoded_input);
  setupWindow(op.getWindowSize(), max_pool_node->window());
  setupStride(op.getStrides(), max_pool_node->stride());
  setupPad(op.getPaddingBefore(), op.getPaddingAfter(), max_pool_node->pad());

  auto *output = _loco_graph->nodes()->create<loco::FeatureDecode>();
  output->input(max_pool_node);
  output->decoder(createFeatureDecoder(op.getDataFormat()));

  _mir2loco_map.emplace(op.getOutput(0), output);
}

void Transformer::visit(mir::ops::MulOp &op)
{
  // Get Input
  loco::Node *lhs = _mir2loco_map.at(op.getInput(0));
  loco::Node *rhs = _mir2loco_map.at(op.getInput(1));
  auto result = createEltwiseBinary<loco::EltwiseMul>(op, lhs, rhs);
  // Not set Shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), result);
}

void Transformer::visit(mir::ops::OutputOp &op)
{
  mir::Operation::Output *mir_input = op.getInput(0);
  loco::Node *input = _mir2loco_map.at(mir_input);

  loco::GraphOutput *graph_output = _loco_graph->outputs()->create();
  graph_output->name(mir_input->getName());
  graph_output->dtype(convertDataType(mir_input->getElementType()));
  graph_output->shape(make_tensor_shape(mir_input->getShape()));

  auto *push_node = _loco_graph->nodes()->create<loco::Push>();
  push_node->from(input);

  loco::link(graph_output, push_node);
}

void Transformer::visit(mir::ops::ReluOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));

  auto relu_node = _loco_graph->nodes()->create<loco::ReLU>();
  relu_node->input(input);
  // Not set shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), relu_node);
}

void Transformer::visit(mir::ops::ReshapeOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));

  auto reshape_node = _loco_graph->nodes()->create<loco::Reshape<loco::ReshapeType::Fixed>>();
  reshape_node->input(input);
  // Set Shape
  auto &out_shape = op.getOutputShape(0);
  setupShape(out_shape, reshape_node);
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), reshape_node);
}

void Transformer::visit(mir::ops::SoftmaxOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));

  auto softmax_node = _loco_graph->nodes()->create<loco::TensorSoftmax>();
  softmax_node->input(input);
  // Set Axis
  softmax_node->axis(op.getAxis());
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), softmax_node);
}

void Transformer::visit(mir::ops::SubOp &op)
{
  // Get Input
  loco::Node *lhs = _mir2loco_map.at(op.getInput(0));
  loco::Node *rhs = _mir2loco_map.at(op.getInput(1));
  auto result = createEltwiseBinary<loco::EltwiseSub>(op, lhs, rhs);
  // Not set Shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), result);
}

void Transformer::visit(mir::ops::TransposeOp &op)
{
  loco::Node *input = _mir2loco_map.at(op.getInput(0));
  const auto &axis_order = op.getAxisOrder();

  auto transpose_node = _loco_graph->nodes()->create<loco::TensorTranspose>();
  transpose_node->input(input);
  // Set axis order
  transpose_node->perm()->size(axis_order.size());
  for (size_t i = 0; i < axis_order.size(); i++)
    transpose_node->perm()->axis(i) = axis_order[i];
  // Not set shape
  // Add to map
  _mir2loco_map.emplace(op.getOutput(0), transpose_node);
}

void Transformer::visit_fallback(mir::Operation &op) { throw std::runtime_error("NYI operation"); }

std::unique_ptr<loco::Graph> Transformer::transform(mir::Graph *mir_graph)
{
  _mir2loco_map.clear();
  _loco_graph.reset();
  _loco_graph = loco::make_graph();

  // Transform Nodes
  mir_graph->accept(this);

  // validate graph
  assert(loco::valid(_loco_graph.get()));

  return std::move(_loco_graph);
}

} // namespace mir2loco
