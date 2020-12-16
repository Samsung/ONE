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

#include "FuseBiasAddPass.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

#include <oops/InternalExn.h>

#include <set>

/*
  Note: Terms for variables in this implementation is as follows:

      ex) subgraph handled:    TFLConv2D -------- TFLAdd
                        (or TFLDepthwiseConv2D)  (or TFLSub)
                                    |                 |
                                   \|/               \|/
            variable name :     former            latter
                Type      :     FormerT           LatterT
                    (shortened name from Mixin)  (template type)
*/
namespace
{

using FormerT = locoex::TFLNodeMixin<locoex::TFLNodeTrait::Bias>;

loco::Node *as_loco_node(FormerT *former)
{
  auto loco_node = dynamic_cast<loco::Node *>(former);
  assert(loco_node != nullptr);

  return loco_node;
}

locoex::TFLConst *get_const(loco::Node *x, loco::Node *y)
{
  if (auto const_node = dynamic_cast<locoex::TFLConst *>(x))
    return const_node;
  else if (auto const_node = dynamic_cast<locoex::TFLConst *>(y))
    return const_node;

  return nullptr;
}

FormerT *get_former(loco::Node *x, loco::Node *y)
{
  if (auto node = dynamic_cast<FormerT *>(x))
    return node;
  else if (auto node = dynamic_cast<FormerT *>(y))
    return node;

  return nullptr;
}

/// @brief Finds input that is TFLConst and set it to new_input
void set_const_input(locoex::TFLNode *node, locoex::TFLConst *new_input)
{
  if (auto add = dynamic_cast<locoex::TFLAdd *>(node))
  {
    if (dynamic_cast<locoex::TFLConst *>(add->x()))
      add->x(new_input);
    else if (dynamic_cast<locoex::TFLConst *>(add->y()))
      add->y(new_input);
    else
      assert(false and "One node should be TFLConst");

    return;
  }

  if (auto sub = dynamic_cast<locoex::TFLSub *>(node))
  {
    if (dynamic_cast<locoex::TFLConst *>(sub->x()))
      sub->x(new_input);
    else if (dynamic_cast<locoex::TFLConst *>(sub->y()))
      sub->y(new_input);
    else
      assert(false and "One node should be TFLConst");

    return;
  }

  assert(false and "Param should be TFLAdd or TFLSub");
}

/**
 * @brief Creates a TFLConst whose shape is [to] and values are all const_node->at(0),
 *        where const_node has only one element(a scalar or a tensor of shape [1])
 */
locoex::TFLConst *create_widened(locoex::TFLConst *const_node, uint32_t to)
{
  auto const_shape = loco::shape_get(const_node).as<loco::TensorShape>();

  assert(const_shape.rank() == 0 or (const_shape.rank() == 1 and const_shape.dim(0) == 1));

  auto g = const_node->graph();

  auto widened_const = g->nodes()->create<locoex::TFLConst>();
  {
    widened_const->dtype(loco::DataType::FLOAT32);
    widened_const->rank(1);
    widened_const->dim(0) = to;
    widened_const->size<loco::DataType::FLOAT32>(to);
    for (uint32_t x = 0; x < to; x++)
      widened_const->at<loco::DataType::FLOAT32>(x) = const_node->at<loco::DataType::FLOAT32>(0);
  }
  return widened_const;
}

template <typename TFLType> float calc(float, float);

template <> float calc<locoex::TFLAdd>(float x, float y) { return x + y; }
template <> float calc<locoex::TFLSub>(float x, float y) { return x - y; }

template <class LatterT> class Fuser
{
public:
  Fuser(LatterT *latter)
  {
    static_assert(std::is_same<LatterT, locoex::TFLAdd>::value ||
                    std::is_same<LatterT, locoex::TFLSub>::value,
                  "wrong template type");

    _latter = latter;
    _graph = _latter->graph();
    _const_node = get_const(_latter->x(), _latter->y());
    _former = get_former(_latter->x(), _latter->y());

    assert(_const_node && _former);
  }

  void fuse(void);

private:
  loco::Graph *_graph;
  LatterT *_latter;
  locoex::TFLConst *_const_node;
  FormerT *_former;

  locoex::TFLConst *create_fused_bias_const();
};

// instantiation
template class Fuser<locoex::TFLAdd>;
template class Fuser<locoex::TFLSub>;

template <class LatterT> locoex::TFLConst *Fuser<LatterT>::create_fused_bias_const()
{
  // we have to create a new bias const by adding/substracting bias and const node (of TFLAdd or
  // TFLSub)
  auto bias = loco::must_cast<locoex::TFLConst *>(_former->bias());
  assert(bias->dtype() == loco::DataType::FLOAT32 &&
         _const_node->dtype() == loco::DataType::FLOAT32);

  assert(bias->rank() == 1 && _const_node->rank() == 1);
  assert(bias->dim(0) == _const_node->dim(0));

  // build a new bias const
  auto new_bias = _graph->nodes()->create<locoex::TFLConst>();
  {
    new_bias->dtype(loco::DataType::FLOAT32);

    new_bias->rank(1);
    new_bias->dim(0) = bias->dim(0);

    new_bias->size<loco::DataType::FLOAT32>(bias->dim(0).value());

    for (uint32_t x = 0; x < bias->dim(0).value(); x++)
      new_bias->at<loco::DataType::FLOAT32>(x) = calc<LatterT>(
        bias->at<loco::DataType::FLOAT32>(x), _const_node->at<loco::DataType::FLOAT32>(x));
  }

  return new_bias;
}

// FuseBiasAddPass works when former->fusedActivationFunction() == NONE
bool check_act_func(FormerT *former)
{
  using FusedActFuncMixin = locoex::TFLNodeMixin<locoex::TFLNodeTrait::FusedActFunc>;

  if (auto node = dynamic_cast<FusedActFuncMixin *>(former))
    return node->fusedActivationFunction() == locoex::FusedActFunc::NONE;
  else
    return true;
}

template <class LatterT> void set_act_func(FormerT *former, LatterT *latter)
{
  using FusedActFuncMixin = locoex::TFLNodeMixin<locoex::TFLNodeTrait::FusedActFunc>;

  if (auto node = dynamic_cast<FusedActFuncMixin *>(former))
    node->fusedActivationFunction(latter->fusedActivationFunction());
}

// instantiation
template void set_act_func(FormerT *, locoex::TFLAdd *);
template void set_act_func(FormerT *, locoex::TFLSub *);

/**
 * @brief Fuse TFLAdd or TFLSub (latter) into TFLConv2d or TFLDepthwiseConv2D (former).
 *        All conditions should be checked before calling this.
 *
 * @note  TFLAdd can have fused activation function (let's call this FAF for simplicity).
 *
 *        Conv2D's FAF    | TFLAdd's FAF        => FAF after fusing TFLAdd into TFLConv2D
 *        ----------------|---------------      --------------------------------------
 *        NONE            | NONE, RELU or RELU6 => TFLAdd's FAF
 *        other than NONE | anything            => cannot be fused
 */
template <class LatterT> void Fuser<LatterT>::fuse(void)
{
  // check fused activation function
  {
    assert(check_act_func(_former));

    set_act_func<LatterT>(_former, _latter);
  }

  auto new_bias = create_fused_bias_const();

  // replace node with new_bias
  // note that loco::replace() is not used because bias could be input of other op just in case
  _former->bias(new_bias);

  // remove TFLAdd or TFLSub node
  loco::replace(_latter).with(as_loco_node(_former));
  _latter->x(nullptr);
  _latter->y(nullptr);
}

struct Collector final : public locoex::TFLNodeMutableVisitor<void>
{
  template <class LatterT>
  void setCandidate(FormerT *former, LatterT *latter, locoex::TFLConst *const_node)
  {
    static_assert(std::is_same<LatterT, locoex::TFLAdd>::value ||
                    std::is_same<LatterT, locoex::TFLSub>::value,
                  "wrong template type");

    if (!check_act_func(former))
      return;

    auto depth =
      loco::shape_get(as_loco_node(former)).template as<loco::TensorShape>().dim(3).value();
    auto const_shape = loco::shape_get(const_node).template as<loco::TensorShape>();

    if (const_shape.rank() == 1 and const_shape.dim(0) == depth)
    {
      candidates.insert(latter);
    }
    // when Const has only one value, create a new const with shape [depth]
    else if (const_shape.rank() == 0 or (const_shape.rank() == 1 and const_shape.dim(0) == 1))
    {
      if (!(loco::dtype_get(as_loco_node(former)) == loco::DataType::FLOAT32))
        INTERNAL_EXN_V("Unsupported data type",
                       oops::to_uint32(loco::dtype_get(as_loco_node(former))));
      if (!(const_node->dtype() == loco::DataType::FLOAT32))
        INTERNAL_EXN_V("Unsupported data type", oops::to_uint32(const_node->dtype()));

      auto new_bias_node = create_widened(const_node, depth);

      // Replacing TFLConst input of TFLAdd or TFLSub.
      // Note that calling loco::replace(const_node).with(new_bias_node) could be dangerous
      // because const_node could be the input of many nodes
      set_const_input(latter, new_bias_node);

      candidates.insert(latter);
    }
  }

  void visit(locoex::TFLAdd *latter) final
  {
    auto former = get_former(latter->x(), latter->y());
    auto const_node = get_const(latter->x(), latter->y());

    if (former && const_node)
      setCandidate<locoex::TFLAdd>(former, latter, const_node);
  }

  void visit(locoex::TFLSub *latter) final
  {
    // TFLSub, of which x() = TFLConv2D or TFLDepthwiseConv2D, y() = TFLConst, is fusing target
    auto former = dynamic_cast<FormerT *>(latter->x());
    auto const_node = dynamic_cast<locoex::TFLConst *>(latter->y());

    if (former && const_node)
      setCandidate<locoex::TFLSub>(former, latter, const_node);
  }

  void visit(locoex::TFLNode *) final { return; }

  std::set<locoex::TFLNode *> candidates;
};

struct Performer final : public locoex::TFLNodeMutableVisitor<void>
{
  void visit(locoex::TFLAdd *latter) final
  {
    assert(get_former(latter->x(), latter->y()));

    Fuser<locoex::TFLAdd> fuser(latter);
    fuser.fuse();
  }

  void visit(locoex::TFLSub *latter) final
  {
    assert(get_former(latter->x(), latter->y()));

    Fuser<locoex::TFLSub> fuser(latter);
    fuser.fuse();
  }

  void visit(locoex::TFLNode *) final { assert(false && "should not be called"); }
};

} // namespace

namespace exo
{

bool FuseBiasAddPass::run(loco::Graph *g)
{
  Collector collector;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (node->dialect() == locoex::TFLDialect::get())
    {
      auto tfl_node = loco::must_cast<locoex::TFLNode *>(node);
      tfl_node->accept(&collector);
    }
  }

  Performer performer;

  for (auto node : collector.candidates)
  {
    node->accept(&performer);
  }

  return collector.candidates.size() > 0;
}

} // namespace exo
