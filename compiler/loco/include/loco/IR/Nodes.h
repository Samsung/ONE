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

#ifndef __LOCO_IR_NODES_H__
#define __LOCO_IR_NODES_H__

#include "loco/IR/Node.h"
#include "loco/IR/Use.h"
#include "loco/IR/Domain.h"
#include "loco/IR/DataType.h"
#include "loco/IR/DataTypeTraits.h"
#include "loco/IR/Dimension.h"
#include "loco/IR/Window.h"
#include "loco/IR/Stride.h"
#include "loco/IR/Padding2D.h"
#include "loco/IR/PaddingND.h"
#include "loco/IR/TensorAxis.h"
#include "loco/IR/TensorAxisSet.h"
#include "loco/IR/FeatureCodec.h"
#include "loco/IR/FilterCodec.h"
#include "loco/IR/DepthwiseFilterCodec.h"
#include "loco/IR/MatrixCodec.h"
#include "loco/IR/NodeMixins.h"
#include "loco/IR/CanonicalNodeDecl.h"
#include "loco/IR/GraphInputIndex.h"
#include "loco/IR/GraphOutputIndex.h"

namespace loco
{

class Graph;
class GraphInput;
class GraphOutput;

/**
 * @brief Make a value visible to user
 */
class Push /* to user */ final
    : public CanonicalNodeDef<CanonicalOpcode::Push, FixedArity<1>::Mixin>
{
public:
  Push() = default;

public:
  Node *from(void) const { return at(0)->node(); }
  void from(Node *node) { at(0)->node(node); }

public:
  void index(const GraphOutputIndex &index);

  /**
   * @brief Get associated output index
   *
   * The behavior of this method is undefined when "index" is not set before.
   *
   * NOTE This method intentionally returns "GraphOutputIndex" instead of "const GraphOutputIndex &"
   *      not to expose the internal implementation details.
   */
  GraphOutputIndex index(void) const;

  /**
   * @brief Check whether index is initialized
   *
   * NOTE "indexed" method does not validate whether index is in a valid range
   */
  bool indexed(void) const { return _index != -1; }

private:
  int64_t _index = -1; // Uninitialized
};

void link(GraphOutput *, Push *push);

/// @brief Find a Push node with a given output index
Push *push_node(Graph *g, const GraphOutputIndex &index);

/**
 * @brief Create a value from user data
 */
class Pull /* from user */ final
    : public CanonicalNodeDef<CanonicalOpcode::Pull, FixedArity<0>::Mixin,
                              With<NodeTrait::TensorShape>::Mixin>
{
public:
  Pull() = default;

public:
  void index(const GraphInputIndex &index);

  /**
   * @brief Get associated input index
   *
   * The behavior of this method is undefined when "index" is not set before.
   *
   * NOTE This method intentionally returns "GraphInputIndex" instead of "const GraphInputIndex &"
   *      not to expose the internal implementation details.
   */
  GraphInputIndex index(void) const;

  /**
   * @brief Check whether index is initialized
   *
   * NOTE "indexed" method does not validate whether index is in a valid range
   */
  bool indexed(void) const { return _index != -1; }

public:
  void dtype(const DataType &d);
  DataType dtype(void) const;

private:
  int64_t _index = -1; // Uninitialized

  /**
   * @brief Locally cached data type attribute
   *
   * TODO Remove this cache once all the clients are updated
   */
  DataType _dtype = DataType::Unknown;
};

void link(GraphInput *, Pull *pull);

/// @brief Find a Pull node with a given input index
Pull *pull_node(Graph *g, const GraphInputIndex &index);

/**
 * @brief Create a new value identical to its input
 *
 * This node may encode memory transfer (such as CPU -> GPU or GPU -> CPU)
 */
class Forward final : public CanonicalNodeDef<CanonicalOpcode::Forward, FixedArity<1>::Mixin>
{
public:
  Forward() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Create a new value that rectifies its input
 */
class ReLU final : public CanonicalNodeDef<CanonicalOpcode::ReLU, FixedArity<1>::Mixin>
{
public:
  ReLU() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Create a new value that rectifies its input capping the units at 6.
 */
class ReLU6 final : public CanonicalNodeDef<CanonicalOpcode::ReLU6, FixedArity<1>::Mixin>
{
public:
  ReLU6() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Create a new value that rectifies its input by tanh
 */
class Tanh final : public CanonicalNodeDef<CanonicalOpcode::Tanh, FixedArity<1>::Mixin>
{
public:
  Tanh() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Create a value from constant byte array
 *
 * @note ConstGen assumes "lexical memory layout".
 *
 * Let us assume that a 'ConstGen' generates a constant tensor of shape "S".
 * for each valid index I, the corresponding value comes from offset(S, I)
 * where the implementation of "offset" is given as follows:
 *
 * uint32_t stride(TensorShape shape, uint32_t axis) {
 *   uint32_t res = 1;
 *   for (uint32_t n = rank(shape) - 1; n > axis; --n) { res *= shape.dim(n); }
 *   return res;
 * }
 *
 * uint32_t offset(TensorShape shape, TensorIndex index) {
 *   uint32_t res = 0;
 *   for (uint32_t n = 0; n < rank(shape); ++n) { res += index.at(n) * stride(shape, n); }
 *   return res;
 * }
 */
class ConstGen final
    : public CanonicalNodeDef<CanonicalOpcode::ConstGen, FixedArity<0>::Mixin,
                              With<NodeTrait::DataType>::Mixin, With<NodeTrait::TensorShape>::Mixin>
{
public:
  ConstGen() = default;

public:
  /**
   * @brief Return the number of reserved elements
   * @note This method returns the number of ELEMENT (not BYTE).
   */
  template <DataType DT> uint32_t size(void) const;

  /**
   * @brief Adjust the number of reserved elements
   */
  template <DataType DT> void size(uint32_t size);

  /**
   * @brief Get the element at a given position
   * @require at(n) is valid only when n < size()
   */
  template <DataType DT> const typename DataTypeImpl<DT>::Type &at(uint32_t n) const;

  /**
   * @brief Update the element at a given position
   * @require at(n) is valid only when n < size()
   */
  template <DataType DT> typename DataTypeImpl<DT>::Type &at(uint32_t n);

private:
  /// @brief Data
  std::vector<uint8_t> _data;
};

/**
 * @brief 2D Max Pooling
 *
 * MaxPool2D takes as input a feature map, and produces another feature map
 *
 * ---
 * Any valid MaxPool2D nodes SHOULD satisfy the following conditions.
 *
 * Let us define several helper functions that takes a MaxPool2D nodes first:
 * - IFM_DOMAIN returns the domain of its input
 * - IFM_H returns the height of its input.
 * - IFM_W returns the width of its input.
 * - PAD_T returns the top padding required over its input
 * - PAD_B returns the bottom padding required over its input
 * - PAD_L returns the left padding required over its input
 * - PAD_R returns the right padding required over its input
 * - WIN_H returns the height of its receptive field.
 * - WIN_W returns the width of its receptive field.
 * - STRIDE_H returns the vertical(= on height) stride.
 * - STRIDE_W returns the horizontal(= on width) stride.
 *
 * Condition 1
 *   Statement
 *
 *   A valid MaxPool2D node M SHOULD satisfy the following condition:
 *   - IFM_DOMAIN(M) == Feature
 *
 *   Motivation
 *
 *   There are many possible ways to encode a feature map as a tensor.
 *   - e.g. NCHW/NHWC/...
 *
 *   In order to give some freedom on memory layout to backend, loco requires a feature map
 *   value to be explicitly encoded via FeatureEncode.
 *
 * Condition 2:
 *   Statement
 *
 *   A valid MaxPool2D node M SHOULD satisfy the following conditions:
 *   - (IFM_H(M) + PAD_T(M) + PAD_B(M) - WIN_H(M)) % STRIDE_H(M) == 0
 *   - (IFM_W(M) + PAD_L(M) + PAD_R(M) - WIN_W(M)) % STRIDE_W(M) == 0
 *
 *   Motivation
 *
 *   The output shape may differ for each NN framework when these conditions do not hold.
 *
 *   In order to mitigate such a difference among NN frameworks, loco requires these conditions
 *   for MaxPool2D nodes.
 *
 *   This means that each frontend implementation SHOULD insert appropriate padding/trimming node
 *   before/after MaxPool2D node according to the semantics of the corresponding NN framework.
 * ---
 */
class MaxPool2D final : public CanonicalNodeDef<CanonicalOpcode::MaxPool2D, FixedArity<1>::Mixin>
{
public:
  Node *ifm(void) const { return at(0)->node(); }
  void ifm(Node *node) { at(0)->node(node); }

public:
  const Padding2D *pad(void) const { return &_pad; }
  Padding2D *pad(void) { return &_pad; }

public:
  const Window<2> *window(void) const { return &_window; }
  Window<2> *window(void) { return &_window; }

public:
  const Stride<2> *stride(void) const { return &_stride; }
  Stride<2> *stride(void) { return &_stride; }

private:
  // Pad
  Padding2D _pad;
  // Window
  Window<2> _window;
  // Stride
  Stride<2> _stride;
};

/**
 * @brief 2D Average Pooling
 *
 * @note Follows MaxPool2D (TODO: describe difference)
 */
class AvgPool2D final : public CanonicalNodeDef<CanonicalOpcode::AvgPool2D, FixedArity<1>::Mixin>
{
public:
  enum class Convention
  {
    Unknown,
    // Use the number of elements in each receptive field as a divisor
    Full,
    // Use the number of valid (non-padding) elements in each receptive field as a divisor
    Valid
  };

public:
  Node *ifm(void) const { return at(0)->node(); }
  void ifm(Node *node) { at(0)->node(node); }

public:
  Convention convention(void) const { return _convention; }
  void convention(const Convention &convention) { _convention = convention; }

public:
  const Padding2D *pad(void) const { return &_pad; }
  Padding2D *pad(void) { return &_pad; }

public:
  const Window<2> *window(void) const { return &_window; }
  Window<2> *window(void) { return &_window; }

public:
  const Stride<2> *stride(void) const { return &_stride; }
  Stride<2> *stride(void) { return &_stride; }

private:
  Convention _convention = Convention::Unknown;
  Padding2D _pad;
  Window<2> _window;
  Stride<2> _stride;
};

/**
 * @brief Create a feature map from a tensor
 */
class FeatureEncode final
    : public CanonicalNodeDef<CanonicalOpcode::FeatureEncode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  FeatureEncoder *encoder(void) const { return _enc.get(); }
  void encoder(std::unique_ptr<FeatureEncoder> &&enc) { _enc = std::move(enc); }

private:
  /// @note "encoder" is mandatory
  std::unique_ptr<FeatureEncoder> _enc{nullptr};
};

/**
 * @brief Create a tensor from a feature map
 */
class FeatureDecode final
    : public CanonicalNodeDef<CanonicalOpcode::FeatureDecode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  FeatureDecoder *decoder(void) const { return _dec.get(); }
  void decoder(std::unique_ptr<FeatureDecoder> &&dec) { _dec = std::move(dec); }

private:
  /// @NOTE "decoder" is mandatory
  std::unique_ptr<FeatureDecoder> _dec{nullptr};
};

/**
 * @brief Create a filter from a tensor
 */
class FilterEncode final
    : public CanonicalNodeDef<CanonicalOpcode::FilterEncode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  FilterEncoder *encoder(void) const { return _enc.get(); }
  void encoder(std::unique_ptr<FilterEncoder> &&enc) { _enc = std::move(enc); }

private:
  /// @note "encoder" is mandatory
  std::unique_ptr<FilterEncoder> _enc{nullptr};
};

/**
 * @brief Create a tensor from a filter
 */
class FilterDecode final
    : public CanonicalNodeDef<CanonicalOpcode::FilterDecode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  FilterDecoder *decoder(void) const { return _dec.get(); }
  void decoder(std::unique_ptr<FilterDecoder> &&dec) { _dec = std::move(dec); }

private:
  /// @note "decoder" is mandatory
  std::unique_ptr<FilterDecoder> _dec{nullptr};
};

/**
 * @brief Create a depthwise filter from a tensor
 */
class DepthwiseFilterEncode final
    : public CanonicalNodeDef<CanonicalOpcode::DepthwiseFilterEncode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  DepthwiseFilterEncoder *encoder(void) const { return _enc.get(); }
  void encoder(std::unique_ptr<DepthwiseFilterEncoder> &&enc) { _enc = std::move(enc); }

private:
  /// @note "encoder" is mandatory
  std::unique_ptr<DepthwiseFilterEncoder> _enc{nullptr};
};

/**
 * @brief Create a tensor from a depthwise filter
 */
class DepthwiseFilterDecode final
    : public CanonicalNodeDef<CanonicalOpcode::DepthwiseFilterDecode, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  DepthwiseFilterDecoder *decoder(void) const { return _dec.get(); }
  void decoder(std::unique_ptr<DepthwiseFilterDecoder> &&dec) { _dec = std::move(dec); }

private:
  /// @note "decoder" is mandatory
  std::unique_ptr<DepthwiseFilterDecoder> _dec{nullptr};
};

enum class ReshapeType
{
  Fixed, // shape is known at compile time
  // Add another type for a case when shape is not known at compile time
};

template <ReshapeType RT> class Reshape;

/**
 * @brief Reshape a tensor to another tensor whose shape is known at compile time
 *
 * @note This class reshapes the shape of an input tensor to _shape.
 *       Each dimension of _shape should be known at compile time.
 *       Any dimension of _shape should be greater than 0.
 *
 *       Interpreter or runtime should lexicographically copy an input tensor into an output tensor.
 *       For example, values of an input tesor of shape [2, 2, 2, 2] will be copied into an output
 *       tensor of new shape [4, 4] like the following:
 *         input[0, 0, 0, 0] => output [0, 0]
 *         input[0, 0, 0, 1] => output [0, 1]
 *         input[0, 0, 1, 0] => output [0, 2]
 *         ...
 *         input[1, 1, 1, 1] => output [3, 3]
 */
template <>
class Reshape<ReshapeType::Fixed> final
    : public CanonicalNodeDef<CanonicalOpcode::FixedReshape, FixedArity<1>::Mixin,
                              With<NodeTrait::TensorShape>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

using FixedReshape = Reshape<ReshapeType::Fixed>;

/**
 * @brief Concatenate two tensors
 *
 * Given an axis, TensorConcat takes as input two tensors and produces a tensor
 * concatenated along the given axis.
 */
class TensorConcat final
    : public CanonicalNodeDef<CanonicalOpcode::TensorConcat, FixedArity<2>::Mixin>
{
public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { at(1)->node(node); }

public:
  uint32_t axis(void) const { return _axis; }
  void axis(uint32_t val) { _axis = val; }

private:
  // Axis
  uint32_t _axis{0};
};

/**
 * @brief 2D Spatial Convolution
 */
class Conv2D final : public CanonicalNodeDef<CanonicalOpcode::Conv2D, FixedArity<2>::Mixin>
{
public:
  Node *ifm(void) const { return at(0)->node(); }
  void ifm(Node *node) { at(0)->node(node); }

  Node *ker(void) const { return at(1)->node(); }
  void ker(Node *node) { at(1)->node(node); }

public:
  const Padding2D *pad(void) const { return &_pad; }
  Padding2D *pad(void) { return &_pad; }

public:
  const Stride<2> *stride(void) const { return &_stride; }
  Stride<2> *stride(void) { return &_stride; }

private:
  Padding2D _pad;
  Stride<2> _stride;

  // TODO Support "Dilation"
};

/**
 * @brief Depthwise 2D Convolution
 */
class DepthwiseConv2D final
    : public CanonicalNodeDef<CanonicalOpcode::DepthwiseConv2D, FixedArity<2>::Mixin>
{
public:
  Node *ifm(void) const { return at(0)->node(); }
  void ifm(Node *node) { at(0)->node(node); }

  Node *ker(void) const { return at(1)->node(); }
  void ker(Node *node) { at(1)->node(node); }

public:
  const Padding2D *pad(void) const { return &_pad; }
  Padding2D *pad(void) { return &_pad; }

public:
  const Stride<2> *stride(void) const { return &_stride; }
  Stride<2> *stride(void) { return &_stride; }

private:
  Padding2D _pad;
  Stride<2> _stride;

  // TODO Support "Dilation"
};

/**
 * @brief Reduce type functions
 */
enum class ReduceFunc
{
  Mean, // ReduceMean
  // TODO Support other reduce operations
};

/**
 * @brief Computes ReduceFunc operations for Tensor domain
 * @note  All the reduce functions always keep dimensions
 */
class TensorReduce final
    : public CanonicalNodeDef<CanonicalOpcode::TensorReduce, FixedArity<1>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  const TensorAxisSet *axes(void) const { return &_axes; }
  TensorAxisSet *axes(void) { return &_axes; }

public:
  ReduceFunc func(void) const { return _func; }
  void func(ReduceFunc func) { _func = func; }

private:
  TensorAxisSet _axes;
  ReduceFunc _func{ReduceFunc::Mean};
};

/**
 * @brief 2D Transposed Convolution
 *
 * @note  TransposedConv2D have a few important conventions that IR users should
 *        understand and follow, so please check below notice carefully.
 *
 *
 * 1. What is 'input' and 'output'
 *
 * For loco canonical TransposedConv2D, 'input' and 'output' mean actual input
 * and output node of TransposedConv2D node. Be careful that some other
 * frameworks may use opposite sense, especially TensorFlow which is inspired by
 * backpropagation of convolution.
 * For example, loco::TransposedConv2D::ifm() means actual input feature map
 * node that is sourced into TransposedConv2D.
 *
 * 2. How to read kernel representation
 *
 * TransposedConv2D::ker() should be a node of Filter domain. Following is what
 * each FilterAxis means as a kernel of TransposedConv2D:
 *   - FilterAxis::Height : kernel's height
 *   - FilterAxis::Width  : kernel's width
 *   - FilterAxis::Depth  : IFM's channel depth
 *   - FilterAxis::Count  : OFM's channel depth
 * TODO We may refactor FilterAxis as follow to reduce ambiguity:
 *   - FilterAxis::Height -> FilterAxis::H
 *   - FilterAxis::Width  -> FilterAxis::W
 *   - FilterAxis::Depth  -> FilterAxis::I
 *   - FilterAxis::Count  -> FilterAxis::O
 *
 *
 * 3. Tight fit rule
 *
 * TransposedConv2D have no information about its output shape. Instead, it
 * always satisfy following 'tight fit' rule for horizontal and vertical
 * dimension:
 *
 *   O = S * ( I - 1 ) + F - P
 *
 *   where
 *     O: output size
 *     S: stride
 *     I: input size
 *     F: effective kernal(filter) size
 *     P: whole pad size (= front + rear pad)
 *
 * With this, output shape is uniquely determined by all inputs and attributes.
 */
class TransposedConv2D final
    : public CanonicalNodeDef<CanonicalOpcode::TransposedConv2D, FixedArity<2>::Mixin>
{
public:
  Node *ifm(void) const { return at(0)->node(); }
  void ifm(Node *node) { at(0)->node(node); }

  Node *ker(void) const { return at(1)->node(); }
  void ker(Node *node) { at(1)->node(node); }

public:
  const Padding2D *pad(void) const { return &_pad; }
  Padding2D *pad(void) { return &_pad; }

public:
  const Stride<2> *stride(void) const { return &_stride; }
  Stride<2> *stride(void) { return &_stride; }

private:
  Padding2D _pad;
  Stride<2> _stride;

  // TODO Support "Dilation"
};

/**
 * @brief Computes softmax activations
 */
template <Domain D> class Softmax;

/**
 * @brief Computes softmax activations for Tensor domain
 */
template <>
class Softmax<Domain::Tensor> final
    : public CanonicalNodeDef<CanonicalOpcode::TensorSoftmax, FixedArity<1>::Mixin>
{
public:
  Softmax() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { return at(0)->node(node); }

  uint32_t axis(void) const { return _axis; }
  void axis(uint32_t axis) { _axis = axis; }

private:
  uint32_t _axis = 0;
};

using TensorSoftmax = Softmax<Domain::Tensor>;

/**
 * @brief Create a "Tensor" from a "Bias"
 */
class BiasDecode final : public CanonicalNodeDef<CanonicalOpcode::BiasDecode, FixedArity<1>::Mixin>
{
public:
  BiasDecode() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Create a "Bias" from a "Tensor"
 *
 * BiasEncode currently requires a rank-1 tensor as its input.
 */
class BiasEncode final : public CanonicalNodeDef<CanonicalOpcode::BiasEncode, FixedArity<1>::Mixin>
{
public:
  BiasEncode() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Produce a value of domain D from an input value (of domain D) and a bias
 */
template <Domain D> class BiasAdd;

/**
 * @brief Add Tensor and Bias
 *
 * for each valid tensor index I
 *   out(I) = value(I) + bias(I.at(axis))
 */
template <>
class BiasAdd<Domain::Tensor> final
    : public CanonicalNodeDef<CanonicalOpcode::TensorBiasAdd, FixedArity<2>::Mixin>
{
public:
  BiasAdd() = default;

public:
  Node *value(void) const { return at(0)->node(); }
  void value(Node *node) { return at(0)->node(node); }

  Node *bias(void) const { return at(1)->node(); }
  void bias(Node *node) { return at(1)->node(node); }

  uint32_t axis(void) const { return _axis; }
  void axis(uint32_t axis) { _axis = axis; }

private:
  uint32_t _axis = 0;
};

//
// Alias for external users
//
// loco::TensorBiasAdd
//        vs.
// loco::BiasAdd<loco::Domain::Tensor>
//
using TensorBiasAdd = BiasAdd<Domain::Tensor>;

/**
 * @brief Add Feature and Bias along "depth" axis
 *
 * for each valid feature index (b, ch, row, col)
 *   out(b, ch, row, col) = value(b, ch, row, col) + bias(ch)
 */
template <>
class BiasAdd<Domain::Feature> final
    : public CanonicalNodeDef<CanonicalOpcode::FeatureBiasAdd, FixedArity<2>::Mixin>
{
public:
  BiasAdd() = default;

public:
  Node *value(void) const { return at(0)->node(); }
  void value(Node *node) { return at(0)->node(node); }

  Node *bias(void) const { return at(1)->node(); }
  void bias(Node *node) { return at(1)->node(node); }
};

using FeatureBiasAdd = BiasAdd<Domain::Feature>;

/**
 * @brief Pads a tensor with constant value
 *
 * Pads a input tensor according to the padding with constant value.
 *
 * The dimension of each axis n of the output is
 * output.dim(n) = padding.front(n) + input.dim(n) + padding.back(n)
 *
 * For example, input tensor of shape [1, 2] with
 *
 * padding.front(0) = 1;
 * padding.back(0) = 2;
 *
 * padding.front(1) = 3;
 * padding.back(1) = 4;
 *
 * will be a output tensor of shape
 * [padding.front(0) + 1 + padding.back(0), padding.front(1) + 2 + padding.back(1)] = [4,9].
 */
class TensorConstantPad final
    : public CanonicalNodeDef<CanonicalOpcode::TensorConstantPad, FixedArity<2>::Mixin>
{
public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

  Node *constant(void) const { return at(1)->node(); }
  void constant(Node *node) { at(1)->node(node); }

public:
  const PaddingND *padding(void) const { return &_padding; }
  PaddingND *padding(void) { return &_padding; }

private:
  PaddingND _padding;
};

/**
 * @brief Elementwise Add lhs and rhs
 */
class EltwiseAdd final : public CanonicalNodeDef<CanonicalOpcode::EltwiseAdd, FixedArity<2>::Mixin>
{
public:
  EltwiseAdd() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Elementwise Maximum of lhs and rhs
 *
 * o = (l > r) ? l : r (element-wise)
 */
class EltwiseMax final : public CanonicalNodeDef<CanonicalOpcode::EltwiseMax, FixedArity<2>::Mixin>
{
public:
  EltwiseMax() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Elementwise Mul lhs and rhs
 */
class EltwiseMul final : public CanonicalNodeDef<CanonicalOpcode::EltwiseMul, FixedArity<2>::Mixin>
{
public:
  EltwiseMul() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Elementwise Sub lhs and rhs
 */
class EltwiseSub final : public CanonicalNodeDef<CanonicalOpcode::EltwiseSub, FixedArity<2>::Mixin>
{
public:
  EltwiseSub() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Elementwise Div lhs and rhs
 */
class EltwiseDiv final : public CanonicalNodeDef<CanonicalOpcode::EltwiseDiv, FixedArity<2>::Mixin>
{
public:
  EltwiseDiv() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Elementwise Sqrt of input
 */
class EltwiseSqrt final
    : public CanonicalNodeDef<CanonicalOpcode::EltwiseSqrt, FixedArity<1>::Mixin>
{
public:
  EltwiseSqrt() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

/**
 * @brief Duplicate elements along specified axes
 *
 * TensorBroadcast takes a tensor and produces another tensor with the same rank but HIGHER
 * dimensionality.
 *
 * To create such a tensor. TensorBroadcast duplicates the element along the specified axes.
 *
 * It is possible to control the degree of duplication with a partial map from TensorAxis to
 * Dimension.
 *
 * TODO Explain the constraints (The dimension of inputs for specified axes SHOULD BE 1).
 * TODO Explain the operation semantics
 */
class TensorBroadcast final
    : public CanonicalNodeDef<CanonicalOpcode::TensorBroadcast, FixedArity<1>::Mixin>
{
public:
  TensorBroadcast() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  class Mapping final
  {
  public:
    Mapping() = default;

  public:
    bool defined(const TensorAxis &axis) const;

    const Dimension &dim(const TensorAxis &axis) const;
    Dimension &dim(const TensorAxis &axis);

  private:
    std::map<TensorAxis, Dimension> _content;
  };

  Mapping *mapping(void) { return &_mapping; }
  const Mapping *mapping(void) const { return &_mapping; }

private:
  Mapping _mapping;
};

/**
 * @brief Create Matrix from Tensor
 *
 * MatrixEncode currently requires a rank-2 Tensor as its input.
 */
class MatrixEncode final
    : public CanonicalNodeDef<CanonicalOpcode::MatrixEncode, FixedArity<1>::Mixin>
{
public:
  MatrixEncode() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  MatrixEncoder *encoder(void) const { return _enc.get(); }
  void encoder(std::unique_ptr<MatrixEncoder> &&enc) { _enc = std::move(enc); }

private:
  /// @note "encoder" is mandatory
  std::unique_ptr<MatrixEncoder> _enc{nullptr};
};

/**
 * @brief Create Tensor from Matrix
 *
 * MatrixDecode currently requires a Matrix as its input.
 */
class MatrixDecode final
    : public CanonicalNodeDef<CanonicalOpcode::MatrixDecode, FixedArity<1>::Mixin>
{
public:
  MatrixDecode() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  MatrixDecoder *decoder(void) const { return _dec.get(); }
  void decoder(std::unique_ptr<MatrixDecoder> &&dec) { _dec = std::move(dec); }

private:
  /// @note "decoder" is mandatory
  std::unique_ptr<MatrixDecoder> _dec{nullptr};
};

/**
 * @brief Matrix Multiplication lhs and rhs
 *
 * LHS and RHS must be on Matrix domain
 */
class MatMul final : public CanonicalNodeDef<CanonicalOpcode::MatMul, FixedArity<2>::Mixin>
{
public:
  MatMul() = default;

public:
  Node *lhs(void) const { return at(0)->node(); }
  void lhs(Node *node) { return at(0)->node(node); }

  Node *rhs(void) const { return at(1)->node(); }
  void rhs(Node *node) { return at(1)->node(node); }
};

/**
 * @brief Permute an input
 *
 * In the following case,
 *
 *    output = loco::TensorTranspose(input)
 *
 * perm()->axis(output's axis) = input's axis
 *
 * Input and output belong to tensor domain.
 */
class TensorTranspose final
    : public CanonicalNodeDef<CanonicalOpcode::TensorTranspose, FixedArity<1>::Mixin>
{
public:
  TensorTranspose() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { return at(0)->node(node); }

  class Perm final
  {
  public:
    Perm() = default;

  public:
    uint32_t size() const { return _vals.size(); }
    void size(uint32_t size) { _vals.resize(size); }

    const TensorAxis &axis(TensorAxis n) const { return _vals[n]; }
    TensorAxis &axis(TensorAxis n) { return _vals[n]; }

  private:
    std::vector<TensorAxis> _vals;
  };

  Perm *perm(void) { return &_perm; }
  const Perm *perm(void) const { return &_perm; }

private:
  Perm _perm;
};

} // namespace loco

#endif // __LOCO_IR_NODES_H__
