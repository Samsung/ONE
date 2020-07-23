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

#ifndef __LOCOEX_IR_TFLNODES_H__
#define __LOCOEX_IR_TFLNODES_H__

#include "TFLNodeDecl.h"
#include "TFLOpcode.h"

#include "FusedActFunc.h"
#include "NodeMixins.h"

#include <loco/IR/Node.h>
#include <loco/IR/NodeMixins.h>
#include <loco/IR/DataTypeTraits.h>

#include <locoex/VariadicArityNode.h>

#include <array>

namespace locoex
{

enum class Padding
{
  UNDEFINED, // This is not defined by TFLite. This was added to prevent programming error.
  SAME,
  VALID,
};

class Filter final
{
public:
  Filter() : _w(1), _h(1) {}

  int32_t w() const { return _w; }
  void w(int32_t w) { _w = w; }

  int32_t h() const { return _h; }
  void h(int32_t h) { _h = h; }

private:
  int32_t _w;
  int32_t _h;
};

class Stride final
{
public:
  Stride() : _w(1), _h(1) {}

  int32_t w() const { return _w; }
  void w(int32_t w) { _w = w; }

  int32_t h() const { return _h; }
  void h(int32_t h) { _h = h; }

private:
  int32_t _w;
  int32_t _h;
};

/// @brief enumeration of mixin class
enum class TFLNodeTrait
{
  FusedActFunc,
  Bias
};

template <TFLNodeTrait T> class TFLNodeMixin;

template <> class TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLNodeMixin() = default;

public:
  FusedActFunc fusedActivationFunction() const { return _fused_act_fun; }
  void fusedActivationFunction(FusedActFunc fused_act_fun) { _fused_act_fun = fused_act_fun; }

private:
  FusedActFunc _fused_act_fun = FusedActFunc::UNDEFINED;
};

/**
 * @brief Mixin class for nodes that has a bias input
 */
template <> class TFLNodeMixin<TFLNodeTrait::Bias>
{
public:
  TFLNodeMixin() = default;

public:
  virtual loco::Node *bias(void) const = 0; /// @brief get the input for bias.
  virtual void bias(loco::Node *node) = 0;  /// @brief set the input for bias.
};

/**
 * @brief ADD in TensorFlow Lite
 */
class TFLAdd final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::ADD>>,
                     public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

/**
 * @brief AVERAGE_POOL_2D in TensorFlow Lite
 */
class TFLAveragePool2D final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::AVERAGE_POOL_2D>>,
                               public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLAveragePool2D() : _padding(Padding::UNDEFINED)
  { /* empty */
  }

public:
  loco::Node *value(void) const { return at(0)->node(); }
  void value(loco::Node *node) { at(0)->node(node); }

  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Filter *filter(void) const { return &_filter; }
  Filter *filter(void) { return &_filter; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding;
  Stride _stride;
  Filter _filter;
};

/**
 * @brief CONCATENATION in TensorFlow Lite
 */
class TFLConcatenation final : public VariadicArityNode<TFLNodeImpl<TFLOpcode::CONCATENATION>>,
                               public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLConcatenation(uint32_t arity) : VariadicArityNode<TFLNodeImpl<TFLOpcode::CONCATENATION>>(arity)
  {
    // TODO Support when arity is 0
    assert(arity >= 1);
  }

public:
  uint32_t numValues(void) const { return arity(); }

public:
  Node *values(uint32_t index) const
  {
    assert(index < numValues());
    return at(index)->node();
  }
  void values(uint32_t index, Node *node)
  {
    assert(index < numValues());
    at(index)->node(node);
  }

public:
  uint32_t axis(void) const { return _axis; }
  void axis(uint32_t axis) { _axis = axis; }

private:
  uint32_t _axis{0};
};

/**
 * @brief Class to build tensor data
 * @note  This will not be exported as a specific op
 */
class TFLConst final : public FixedArityNode<0, TFLNodeImpl<TFLOpcode::CONST>>,
                       public loco::NodeMixin<loco::NodeTrait::DataType>,
                       public loco::NodeMixin<loco::NodeTrait::TensorShape>
{
public:
  TFLConst() = default;

public:
  template <loco::DataType DT> uint32_t size(void) const;
  template <loco::DataType DT> void size(uint32_t size);
  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
  template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &at(uint32_t n);

private:
  std::vector<uint8_t> _data;
};

/**
 * @brief CONV_2D in TensorFlow Lite
 */
class TFLConv2D final : public FixedArityNode<3, TFLNodeImpl<TFLOpcode::CONV_2D>>,
                        public TFLNodeMixin<TFLNodeTrait::FusedActFunc>,
                        public TFLNodeMixin<TFLNodeTrait::Bias>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(loco::Node *node) { at(1)->node(node); }

  loco::Node *bias(void) const override { return at(2)->node(); }
  void bias(loco::Node *node) override { at(2)->node(node); }

public:
  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding = Padding::UNDEFINED;
  Stride _stride;
};

/**
 * @brief DEPTHWISE_CONV_2D in TensorFlow Lite
 */
class TFLDepthwiseConv2D final
    : public FixedArityNode<3, TFLNodeImpl<TFLOpcode::DEPTHWISE_CONV_2D>>,
      public TFLNodeMixin<TFLNodeTrait::FusedActFunc>,
      public TFLNodeMixin<TFLNodeTrait::Bias>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(loco::Node *node) { at(1)->node(node); }

  loco::Node *bias(void) const override { return at(2)->node(); }
  void bias(loco::Node *node) override { at(2)->node(node); }

public:
  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

  int32_t depthMultiplier(void) const { return _depth_multiplier; }
  void depthMultiplier(int32_t arg) { _depth_multiplier = arg; }

private:
  Padding _padding = Padding::UNDEFINED;
  Stride _stride;
  int32_t _depth_multiplier = 0;
};

/**
 * @brief DIV in TensorFlow Lite
 */
class TFLDiv final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::DIV>>,
                     public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLDiv() = default;

public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

/**
 * @brief FULLY_CONNECTED in TensorFlow Lite
 */
class TFLFullyConnected final : public FixedArityNode<3, TFLNodeImpl<TFLOpcode::FULLY_CONNECTED>>,
                                public TFLNodeMixin<TFLNodeTrait::FusedActFunc>,
                                public TFLNodeMixin<TFLNodeTrait::Bias>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *weights(void) const { return at(1)->node(); }
  void weights(loco::Node *node) { at(1)->node(node); }

  loco::Node *bias(void) const override { return at(2)->node(); }
  void bias(loco::Node *node) override { at(2)->node(node); }
};

/**
 * @brief MAXIMUM in TensorFlow Lite
 */
class TFLMaximum final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::MAXIMUM>>
{
public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

/**
 * @brief MAX_POOL_2D in TensorFlow Lite
 */
class TFLMaxPool2D final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::MAX_POOL_2D>>,
                           public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLMaxPool2D() : _padding(Padding::UNDEFINED)
  { /* empty */
  }

public:
  loco::Node *value(void) const { return at(0)->node(); }
  void value(loco::Node *node) { at(0)->node(node); }

  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Filter *filter(void) const { return &_filter; }
  Filter *filter(void) { return &_filter; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding;
  Stride _stride;
  Filter _filter;
};

class TFLMean final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::MEAN>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *reduction_indices(void) const { return at(1)->node(); }
  void reduction_indices(loco::Node *node) { at(1)->node(node); }

public:
  bool keep_dims(void) const { return _keep_dims; }
  void keep_dims(bool keep_dims) { _keep_dims = keep_dims; }

private:
  bool _keep_dims = false;
};

/**
 * @brief MUL in TensorFlow Lite
 */
class TFLMul final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::MUL>>,
                     public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

class TFLRelu final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::RELU>>
{
public:
  TFLRelu() = default;

public:
  loco::Node *features(void) const { return at(0)->node(); }
  void features(loco::Node *node) { at(0)->node(node); }
};

class TFLRelu6 final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::RELU6>>
{
public:
  TFLRelu6() = default;

public:
  loco::Node *features(void) const { return at(0)->node(); }
  void features(loco::Node *node) { at(0)->node(node); }
};

class TFLReshape final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::RESHAPE>>
{
public:
  TFLReshape() = default;

public:
  loco::Node *tensor(void) const { return at(0)->node(); }
  void tensor(loco::Node *node) { at(0)->node(node); }

  // TODO Make this input optional. That is, loco system does not emit error
  //      with this input being null
  loco::Node *shape(void) const { return at(1)->node(); }
  void shape(loco::Node *node) { at(1)->node(node); }

public:
  class Shape
  {
  public:
    uint32_t rank(void) const { return _shape.size(); }
    void rank(uint32_t rank) { _shape.resize(rank); }

    int32_t dim(uint32_t n) const { return _shape.at(n); }
    int32_t &dim(uint32_t n) { return _shape.at(n); }

  private:
    std::vector<int32_t> _shape;
  };

  const Shape *newShape(void) const { return &_new_shape; }
  Shape *newShape(void) { return &_new_shape; }

private:
  Shape _new_shape;
};

/**
 * @brief  Set both TFLReshape's 2nd input as TFLConst, and newShape attribute
 *         with same value
 * @note   Shape inference for TFLReshape forces them to be same
 * TODO find better place for this helper
 */
void set_new_shape(locoex::TFLReshape *node, int32_t *base, uint32_t size);

class TFLRsqrt final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::RSQRT>>
{
public:
  TFLRsqrt() = default;

public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }
};

// TODO TFLSoftmax

class TFLSqrt final : public FixedArityNode<1, TFLNodeImpl<TFLOpcode::SQRT>>
{
public:
  TFLSqrt() = default;

public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }
};

class TFLSquaredDifference final
    : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::SQUARED_DIFFERENCE>>
{
public:
  TFLSquaredDifference() = default;

public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

/**
 * @brief SUB in TensorFlow Lite
 */
class TFLSub final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::SUB>>,
                     public TFLNodeMixin<TFLNodeTrait::FusedActFunc>
{
public:
  TFLSub() = default;

public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }
};

// TODO TFLTanh

/**
 * @brief TRANSPOSE in TensorFlow Lite
 */
class TFLTranspose final : public FixedArityNode<2, TFLNodeImpl<TFLOpcode::TRANSPOSE>>
{
public:
  TFLTranspose() = default;

public:
  /// @brief Get the input node to transpose
  loco::Node *a(void) const { return at(0)->node(); }

  /// @brief Set the input node to transpose
  void a(loco::Node *node) { at(0)->node(node); }

  loco::Node *perm(void) const { return at(1)->node(); }
  void perm(loco::Node *node) { at(1)->node(node); }
};

/**
 * @brief TRANSPOSE_CONV in TensorFlow Lite
 *
 * @note  Argument node function names are from TensorFlow. So refering 'in' and
 *        'out' acutally means 'out' and 'in' of the this node.
 */
class TFLTransposeConv final : public FixedArityNode<3, TFLNodeImpl<TFLOpcode::TRANSPOSE_CONV>>
{
public:
  loco::Node *inputSizes(void) const { return at(0)->node(); }
  void inputSizes(Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(Node *node) { at(1)->node(node); }

  loco::Node *outBackprop(void) const { return at(2)->node(); }
  void outBackprop(Node *node) { at(2)->node(node); }

public:
  const Padding &padding(void) const { return _padding; }
  void padding(const Padding &padding) { _padding = padding; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding{Padding::UNDEFINED};
  Stride _stride;
};

// TODO define more children of TFLNode

} // namespace locoex

#endif // __LOCOEX_IR_TFLNODES_H__
