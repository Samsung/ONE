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

#include "loco/Service/CanonicalShapeInferenceRule.h"
#include "loco/Service/ShapeInference.h"

#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>
#include <loco/IR/CanonicalNodeVisitor.h>

#include <cassert>

namespace
{

struct PlaneShape
{
  loco::Dimension height;
  loco::Dimension width;
};

PlaneShape make_plane_shape(const loco::FeatureShape &feature_shape)
{
  PlaneShape plane_shape;

  plane_shape.height = feature_shape.height();
  plane_shape.width = feature_shape.width();

  return plane_shape;
}

class FeatureShapeUpdater final
{
public:
  FeatureShapeUpdater(loco::FeatureShape *ptr) : _feature_shape_ptr{ptr}
  {
    // DO NOTHING
  }

public:
  void with(const PlaneShape &plane_shape) const
  {
    _feature_shape_ptr->height() = plane_shape.height;
    _feature_shape_ptr->width() = plane_shape.width;
  }

private:
  loco::FeatureShape *_feature_shape_ptr;
};

/**
 * HOW TO USE
 *
 *   loco::FeatureShape feature_shape = ...;
 *
 *   update(feature_shape).with(...)
 */
FeatureShapeUpdater update(loco::FeatureShape &feature_shape)
{
  return FeatureShapeUpdater{&feature_shape};
}

loco::Window<2> window_of(const loco::FilterShape &filter_shape)
{
  loco::Window<2> window;

  window.vertical(filter_shape.height().value());
  window.horizontal(filter_shape.width().value());

  return window;
}

loco::Window<2> window_of(const loco::DepthwiseFilterShape &depthwise_filter_shape)
{
  loco::Window<2> window;

  window.vertical(depthwise_filter_shape.height().value());
  window.horizontal(depthwise_filter_shape.width().value());

  return window;
}

enum class Direction
{
  Forward,
  Backward,
};

template <Direction> class PlaneInference;

template <> class PlaneInference<Direction::Forward> final
{
public:
  PlaneShape operator()(const PlaneShape &in) const
  {
    assert(_pad != nullptr);
    assert(_window != nullptr);
    assert(_stride != nullptr);

    uint32_t const raw_input_height = in.height.value();
    uint32_t const raw_input_width = in.width.value();

    uint32_t const raw_window_height = _window->vertical();
    uint32_t const raw_window_width = _window->horizontal();

    uint32_t const vertical_padding = _pad->top() + _pad->bottom();
    uint32_t const horizontal_padding = _pad->left() + _pad->right();

    uint32_t const effective_input_height = raw_input_height + vertical_padding;
    uint32_t const effective_input_width = raw_input_width + horizontal_padding;

    // NOTE To support "dilation" later
    uint32_t const effective_window_height = raw_window_height;
    uint32_t const effective_window_width = raw_window_width;

    uint32_t const vertical_stride = _stride->vertical();
    uint32_t const horizontal_stride = _stride->horizontal();

    assert((effective_input_height - effective_window_height) % vertical_stride == 0);
    assert((effective_input_width - effective_window_width) % horizontal_stride == 0);

    PlaneShape res;

    res.height = (effective_input_height - effective_window_height) / vertical_stride + 1;
    res.width = (effective_input_width - effective_window_width) / horizontal_stride + 1;

    return res;
  }

public:
  void pad(const loco::Padding2D *value) { _pad = value; }
  void window(const loco::Window<2> *value) { _window = value; }
  void stride(const loco::Stride<2> *value) { _stride = value; }

private:
  const loco::Padding2D *_pad = nullptr;
  const loco::Window<2> *_window = nullptr;
  const loco::Stride<2> *_stride = nullptr;
};

template <> class PlaneInference<Direction::Backward> final
{
public:
  PlaneShape operator()(const PlaneShape &in) const
  {
    assert(_pad != nullptr);
    assert(_window != nullptr);
    assert(_stride != nullptr);

    uint32_t const input_height = in.height.value();
    uint32_t const input_width = in.width.value();

    uint32_t const vertical_padding = _pad->top() + _pad->bottom();
    uint32_t const horizontal_padding = _pad->left() + _pad->right();

    uint32_t const raw_window_height = _window->vertical();
    uint32_t const raw_window_width = _window->horizontal();

    // TODO Support "dilation"
    uint32_t const effective_window_height = raw_window_height;
    uint32_t const effective_window_width = raw_window_width;

    uint32_t const vertical_stride = _stride->vertical();
    uint32_t const horizontal_stride = _stride->horizontal();

    PlaneShape res;

    res.height = vertical_stride * (input_height - 1) + effective_window_height - vertical_padding;
    res.width = horizontal_stride * (input_width - 1) + effective_window_width - horizontal_padding;

    return res;
  }

public:
  void pad(const loco::Padding2D *value) { _pad = value; }
  void window(const loco::Window<2> *value) { _window = value; }
  void stride(const loco::Stride<2> *value) { _stride = value; }

private:
  const loco::Padding2D *_pad = nullptr;
  const loco::Window<2> *_window = nullptr;
  const loco::Stride<2> *_stride = nullptr;
};

/**
 * There are two possible maintenance policies.
 * - Introduce a new canonical node first, and then extend this algorithm later
 * - Introduce a new canonical node and extend this algorithm at the same time
 *
 * The current implementation assumes the former one (for historical reason).
 *
 * TODO Evaluate the impact of the latter one
 *
 * NOTE "Forward" means that this algorithm computes the ouput shape from inputs shapes
 */
class ForwardShapeInferenceAlgorithm final : public loco::CanonicalNodeVisitor<loco::NodeShape>
{
public:
  ForwardShapeInferenceAlgorithm(const loco::ShapeInferenceRule::Context *ctx) : _ctx{ctx}
  {
    // DO NOTHING
  }

private:
  const loco::ShapeInferenceRule::Context *_ctx;

private:
  bool shape_known(const loco::Node *node) const { return _ctx->known(node); }
  loco::NodeShape node_shape(const loco::Node *node) const { return _ctx->get(node); }

private:
  loco::NodeShape eltwise_binary_node_shape(const loco::Node *node)
  {
    // This helper works only for binary node.
    assert(node->arity() == 2);

    auto lhs_shape = node_shape(node->arg(0));
    auto rhs_shape = node_shape(node->arg(1));

    // ASSERT: lhs_shape == rhs_shape

    return lhs_shape;
  }

public:
  // CASE: AvgPool2D
  loco::NodeShape visit(const loco::AvgPool2D *node) final
  {
    PlaneInference<Direction::Forward> infer_plane_shape;

    infer_plane_shape.pad(node->pad());
    infer_plane_shape.window(node->window());
    infer_plane_shape.stride(node->stride());

    auto input_feature_shape = node_shape(node->ifm()).as<loco::FeatureShape>();
    auto input_plane_shape = make_plane_shape(input_feature_shape);
    auto output_plane_shape = infer_plane_shape(input_plane_shape);
    auto output_feature_shape = input_feature_shape; // AvgPool2D does not change count/depth

    // Update the height/width of output_feature_shape with that of output_plane_shape
    update(output_feature_shape).with(output_plane_shape);

    return loco::NodeShape{output_feature_shape};
  }

  // CASE: BiasDecode
  loco::NodeShape visit(const loco::BiasDecode *node) final
  {
    // The input of BiasDecode SHOULD BE a bias!
    assert(node_shape(node->input()).domain() == loco::Domain::Bias);
    auto input_bias_shape = node_shape(node->input()).as<loco::BiasShape>();

    loco::TensorShape output_tensor_shape;

    output_tensor_shape.rank(1);
    output_tensor_shape.dim(0) = input_bias_shape.length();

    return loco::NodeShape{output_tensor_shape};
  }

  // CASE: BiasEncode
  loco::NodeShape visit(const loco::BiasEncode *node) final
  {
    // The input of BiasEncode SHOULD BE a tensor!
    assert(node_shape(node->input()).domain() == loco::Domain::Tensor);
    auto input_tensor_shape = node_shape(node->input()).as<loco::TensorShape>();

    loco::BiasShape output_bias_shape;

    output_bias_shape.length() = input_tensor_shape.dim(0);

    return loco::NodeShape{output_bias_shape};
  }

  // CASE: ConstGen
  loco::NodeShape visit(const loco::ConstGen *node) final
  {
    loco::TensorShape tensor_shape;

    tensor_shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); ++axis)
    {
      tensor_shape.dim(axis) = node->dim(axis);
    }

    return loco::NodeShape{tensor_shape};
  }

  // CASE: Conv2D
  loco::NodeShape visit(const loco::Conv2D *node) final
  {
    auto filter_shape = node_shape(node->ker()).as<loco::FilterShape>();
    auto filter_window = window_of(filter_shape);

    PlaneInference<Direction::Forward> infer_plane_shape;

    infer_plane_shape.pad(node->pad());
    infer_plane_shape.window(&filter_window);
    infer_plane_shape.stride(node->stride());

    auto input_feature_shape = node_shape(node->ifm()).as<loco::FeatureShape>();
    auto input_plane_shape = make_plane_shape(input_feature_shape);
    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    loco::FeatureShape output_feature_shape;

    // "COUNT" does not change
    output_feature_shape.count() = input_feature_shape.count();
    // "DEPTH" depends on # of filters
    output_feature_shape.depth() = filter_shape.count();
    // Update the height/width of output_feature_shape with that of output_plane_shape
    update(output_feature_shape).with(output_plane_shape);

    return loco::NodeShape{output_feature_shape};
  }

  // CASE: DepthwiseConv2D
  loco::NodeShape visit(const loco::DepthwiseConv2D *node) final
  {
    auto depthwise_filter_shape = node_shape(node->ker()).as<loco::DepthwiseFilterShape>();
    auto dpethwise_filter_window = window_of(depthwise_filter_shape);

    PlaneInference<Direction::Forward> infer_plane_shape;

    infer_plane_shape.pad(node->pad());
    infer_plane_shape.window(&dpethwise_filter_window);
    infer_plane_shape.stride(node->stride());

    auto input_feature_shape = node_shape(node->ifm()).as<loco::FeatureShape>();
    auto input_plane_shape = make_plane_shape(input_feature_shape);
    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    loco::FeatureShape output_feature_shape;

    // "COUNT" does not change
    output_feature_shape.count() = input_feature_shape.count();
    // "DEPTH" depends on [in_channels * channel_multiplier] of filters
    output_feature_shape.depth() = loco::Dimension(depthwise_filter_shape.depth().value() *
                                                   depthwise_filter_shape.multiplier().value());
    // Update the height/width of output_feature_shape with that of output_plane_shape
    update(output_feature_shape).with(output_plane_shape);

    return loco::NodeShape{output_feature_shape};
  }

  // CASE: DepthwiseFilterEncode
  loco::NodeShape visit(const loco::DepthwiseFilterEncode *node) final
  {
    auto input_tensor_shape = node_shape(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{node->encoder()->shape(input_tensor_shape)};
  }

  // CASE: DepthwiseFilterDecode
  loco::NodeShape visit(const loco::DepthwiseFilterDecode *node) final
  {
    auto input_dw_filter_shape = node_shape(node->input()).as<loco::DepthwiseFilterShape>();
    return loco::NodeShape{node->decoder()->shape(input_dw_filter_shape)};
  }

  // CASE: EltwiseAdd
  loco::NodeShape visit(const loco::EltwiseAdd *node) final
  {
    return eltwise_binary_node_shape(node);
  }

  // CASE: EltwiseDiv
  loco::NodeShape visit(const loco::EltwiseDiv *node) final
  {
    return eltwise_binary_node_shape(node);
  }

  // CASE: EltwiseMax
  loco::NodeShape visit(const loco::EltwiseMax *node) final
  {
    return eltwise_binary_node_shape(node);
  }

  // CASE: EltwiseMul
  loco::NodeShape visit(const loco::EltwiseMul *node) final
  {
    return eltwise_binary_node_shape(node);
  }

  // CASE: EltwiseSqrt
  loco::NodeShape visit(const loco::EltwiseSqrt *node) final { return node_shape(node->input()); }

  // CASE: EltwiseSub
  loco::NodeShape visit(const loco::EltwiseSub *node) final
  {
    return eltwise_binary_node_shape(node);
  }

  // CASE: Forward
  loco::NodeShape visit(const loco::Forward *node) final { return node_shape(node->input()); }

  // CASE: FeatureBiasAdd
  loco::NodeShape visit(const loco::FeatureBiasAdd *node) final
  {
    assert(node_shape(node->value()).domain() == loco::Domain::Feature);
    assert(node_shape(node->bias()).domain() == loco::Domain::Bias);

    // Q. What to do when there is a mismatch between value's depth and bias's length?

    return node_shape(node->value());
  }

  // CASE: FeatureDecode
  loco::NodeShape visit(const loco::FeatureDecode *node) final
  {
    auto input_node_shape = node_shape(node->input());
    return loco::NodeShape{node->decoder()->shape(input_node_shape.as<loco::FeatureShape>())};
  }

  // CASE: FeatureEncode
  loco::NodeShape visit(const loco::FeatureEncode *node) final
  {
    auto input_node_shape = node_shape(node->input());
    return loco::NodeShape{node->encoder()->shape(input_node_shape.as<loco::TensorShape>())};
  }

  // CASE: FilterDecode
  loco::NodeShape visit(const loco::FilterDecode *node) final
  {
    auto input_filter_shape = node_shape(node->input()).as<loco::FilterShape>();
    return loco::NodeShape{node->decoder()->shape(input_filter_shape)};
  }

  // CASE: FilterEncode
  loco::NodeShape visit(const loco::FilterEncode *node) final
  {
    auto input_tensor_shape = node_shape(node->input()).as<loco::TensorShape>();
    return loco::NodeShape{node->encoder()->shape(input_tensor_shape)};
  }

  // CASE: FixedReshape
  loco::NodeShape visit(const loco::FixedReshape *node) final
  {
    loco::TensorShape tensor_shape;

    tensor_shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); ++axis)
    {
      tensor_shape.dim(axis) = node->dim(axis);
    }

    return loco::NodeShape{tensor_shape};
  }

  // CASE: MatMul
  loco::NodeShape visit(const loco::MatMul *node) final
  {
    assert(shape_known(node->lhs()));
    assert(shape_known(node->rhs()));
    auto const lhs_shape = node_shape(node->lhs()).as<loco::MatrixShape>();
    auto const rhs_shape = node_shape(node->rhs()).as<loco::MatrixShape>();

    loco::MatrixShape out_shape;

    // Checking shape capability for multiplication
    assert(lhs_shape.width() == rhs_shape.height());

    out_shape.height() = lhs_shape.height();
    out_shape.width() = rhs_shape.width();

    return out_shape;
  }

  // CASE: MatrixDecode
  loco::NodeShape visit(const loco::MatrixDecode *node) final
  {
    auto input_node_shape = node_shape(node->input());
    return loco::NodeShape{node->decoder()->shape(input_node_shape.as<loco::MatrixShape>())};
  }

  // CASE: MatrixEncode
  loco::NodeShape visit(const loco::MatrixEncode *node) final
  {
    auto input_node_shape = node_shape(node->input());
    return loco::NodeShape{node->encoder()->shape(input_node_shape.as<loco::TensorShape>())};
  }

  // CASE: MaxPool2D
  loco::NodeShape visit(const loco::MaxPool2D *node) final
  {
    PlaneInference<Direction::Forward> infer_plane_shape;

    infer_plane_shape.pad(node->pad());
    infer_plane_shape.window(node->window());
    infer_plane_shape.stride(node->stride());

    auto input_feature_shape = node_shape(node->ifm()).as<loco::FeatureShape>();
    auto input_plane_shape = make_plane_shape(input_feature_shape);
    auto output_plane_shape = infer_plane_shape(input_plane_shape);
    auto output_feature_shape = input_feature_shape; // MaxPool2D does not change count/depth

    // Update the height/width of output_feature_shape with that of output_plane_shape
    update(output_feature_shape).with(output_plane_shape);

    return loco::NodeShape{output_feature_shape};
  }

  // CASE: Push
  loco::NodeShape visit(const loco::Push *node) final
  {
    assert(shape_known(node->from()));
    return node_shape(node->from());
  }

  // CASE: Pull
  loco::NodeShape visit(const loco::Pull *node) final
  {
    // Build a tensor shape from "Pull" node
    loco::TensorShape tensor_shape;

    tensor_shape.rank(node->rank());
    for (uint32_t axis = 0; axis < node->rank(); ++axis)
    {
      tensor_shape.dim(axis) = node->dim(axis);
    }

    return loco::NodeShape{tensor_shape};
  }

  // CASE: ReLU
  loco::NodeShape visit(const loco::ReLU *node) final { return node_shape(node->input()); }

  // CASE: ReLU6
  loco::NodeShape visit(const loco::ReLU6 *node) final { return node_shape(node->input()); }

  // CASE: Tanh
  loco::NodeShape visit(const loco::Tanh *node) final { return node_shape(node->input()); }

  // CASE: TensorBiasAdd
  loco::NodeShape visit(const loco::TensorBiasAdd *node) final
  {
    assert(node_shape(node->value()).domain() == loco::Domain::Tensor);
    assert(node_shape(node->bias()).domain() == loco::Domain::Bias);

    // Q. What to do when there is a mismatch between value's dim and bias's length?

    return node_shape(node->value());
  }

  // CASE: TensorConcat
  loco::NodeShape visit(const loco::TensorConcat *node)
  {
    auto const lhs_shape = node_shape(node->lhs()).as<loco::TensorShape>();
    auto const rhs_shape = node_shape(node->rhs()).as<loco::TensorShape>();

    assert(lhs_shape.rank() == rhs_shape.rank());
    uint32_t const out_rank = lhs_shape.rank();

    loco::TensorShape out_shape;

    out_shape.rank(out_rank);

    for (uint32_t axis = 0; axis < out_rank; ++axis)
    {
      if (axis == node->axis())
      {
        out_shape.dim(axis) = lhs_shape.dim(axis).value() + rhs_shape.dim(axis).value();
      }
      else
      {
        assert(lhs_shape.dim(axis) == rhs_shape.dim(axis));
        out_shape.dim(axis) = lhs_shape.dim(axis);
      }
    }

    return loco::NodeShape{out_shape};
  }

  // CASE: TensorBroadcast
  loco::NodeShape visit(const loco::TensorBroadcast *node) final
  {
    auto tensor_shape = node_shape(node->input()).as<loco::TensorShape>();
    auto const tensor_rank = tensor_shape.rank();

    for (uint32_t axis = 0; axis < tensor_rank; ++axis)
    {
      if (node->mapping()->defined(axis))
      {
        tensor_shape.dim(axis) = node->mapping()->dim(axis);
      }
    }

    return loco::NodeShape{tensor_shape};
  }

  // CASE: TensorReduce
  loco::NodeShape visit(const loco::TensorReduce *node) final
  {
    auto tensor_shape = node_shape(node->input()).as<loco::TensorShape>();
    auto const tensor_rank = tensor_shape.rank();

    for (uint32_t d = 0; d < tensor_rank; ++d)
      if (node->axes()->defined(d))
        tensor_shape.dim(d) = 1;

    return loco::NodeShape{tensor_shape};
  }

  // CASE: TensorSoftmax
  loco::NodeShape visit(const loco::TensorSoftmax *node) final { return node_shape(node->input()); }

  // CASE: TensorTranspose
  loco::NodeShape visit(const loco::TensorTranspose *node) final
  {
    loco::TensorShape output_shape;

    auto input_shape = node_shape(node->input()).as<loco::TensorShape>();
    assert(input_shape.rank() == node->perm()->size());

    output_shape.rank(input_shape.rank());

    for (uint32_t output_axis = 0; output_axis < output_shape.rank(); output_axis++)
    {
      auto new_dim = input_shape.dim(node->perm()->axis(output_axis));
      output_shape.dim(output_axis) = new_dim;
    }

    return loco::NodeShape(output_shape);
  }

  // CASE: TransposedConv2D
  loco::NodeShape visit(const loco::TransposedConv2D *node) final
  {
    auto filter_shape = node_shape(node->ker()).as<loco::FilterShape>();
    auto filter_window = window_of(filter_shape);

    PlaneInference<Direction::Backward> infer_plane_shape;

    infer_plane_shape.pad(node->pad());
    infer_plane_shape.window(&filter_window);
    infer_plane_shape.stride(node->stride());

    auto input_feature_shape = node_shape(node->ifm()).as<loco::FeatureShape>();
    auto input_plane_shape = make_plane_shape(input_feature_shape);
    auto output_plane_shape = infer_plane_shape(input_plane_shape);

    loco::FeatureShape output_feature_shape;

    // "COUNT" does not change
    output_feature_shape.count() = input_feature_shape.count();
    // Output "DEPTH" depends on count of filters
    output_feature_shape.depth() = filter_shape.count();
    // Update the height/width of output_feature_shape with that of output_plane_shape
    update(output_feature_shape).with(output_plane_shape);

    return loco::NodeShape{output_feature_shape};
  }

  // CASE: TensorConstantPad
  loco::NodeShape visit(const loco::TensorConstantPad *node) final
  {
    auto const tensor_shape = loco::shape_get(node->input()).as<loco::TensorShape>();
    auto padding = node->padding();

    loco::TensorShape out_shape;
    out_shape.rank(tensor_shape.rank());
    for (uint32_t axis = 0; axis < out_shape.rank(); ++axis)
    {
      out_shape.dim(axis) =
          tensor_shape.dim(axis).value() + padding->front(axis) + padding->back(axis);
    }

    return loco::NodeShape{out_shape};
  }
};

} // namespace

namespace
{
namespace compat
{

struct Context final : public loco::ShapeInferenceRule::Context
{
  bool known(const loco::Node *node) const final { return loco::shape_known(node); }
  loco::NodeShape get(const loco::Node *node) const final { return loco::shape_get(node); }
};

class Sink final : public loco::ShapeInferenceRule::Sink
{
public:
  enum Status
  {
    Unknown,
    Okay,
    Fail,
  };

public:
  const Status &status(void) const { return _status; }
  const loco::NodeShape &shape(void) const { return _shape; }

public:
  void okay(const loco::NodeShape &shape) final
  {
    _status = Okay;
    _shape = shape;
  }

  void fail(void) final
  {
    // Notify failure
    _status = Fail;
  }

private:
  Status _status = Unknown;
  loco::NodeShape _shape;
};

} // namespace compat
} // namespace

namespace loco
{

bool CanonicalShapeInferenceRule::support(const API &api) const
{
  return api == API::V1 or api == API::V2;
}

bool CanonicalShapeInferenceRule::recognize(const Dialect *d) const
{
  return CanonicalDialect::get() == d;
}

bool CanonicalShapeInferenceRule::infer(const Node *node, NodeShape &shape) const
{
  ::compat::Context ctx;
  ::compat::Sink sink;

  infer(&ctx, node, &sink);

  assert(sink.status() == ::compat::Sink::Okay or sink.status() == ::compat::Sink::Fail);

  if (sink.status() == ::compat::Sink::Fail)
  {
    return false;
  }

  shape = sink.shape();
  return true;
}

void CanonicalShapeInferenceRule::infer(const Context *ctx, const Node *node, Sink *sink) const
{
  assert(node->dialect() == loco::CanonicalDialect::get());
  assert(dynamic_cast<const loco::CanonicalNode *>(node) != nullptr);

  ForwardShapeInferenceAlgorithm alg{ctx};
  auto shape = loco::must_cast<const loco::CanonicalNode *>(node)->accept(&alg);

  sink->okay(shape);
}

} // namespace loco
