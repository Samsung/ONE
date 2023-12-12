/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseGRUPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cmath>

#include <cassert>

// Helper to fuse GRU
namespace
{

class GRUPatternBase
{
public:
  GRUPatternBase(luci::CircleNode *candidate) { _pattern_last_node = candidate; }

  virtual ~GRUPatternBase() = default;

public:
  virtual bool matched() = 0;

public:
  luci::CircleNode *_ifm = nullptr;
  luci::CircleConst *_weight_ih = nullptr;
  luci::CircleConst *_bias_ih = nullptr;
  luci::CircleConst *_weight_hh = nullptr;
  luci::CircleConst *_bias_hh = nullptr;

  luci::CircleConst *_hidden_input = nullptr;

  luci::CircleConst *_less_const = nullptr;

  luci::CircleWhile *_while_node = nullptr;
  luci::CircleWhileOut *_while_out_node = nullptr;
  luci::CircleNode *_pattern_last_node = nullptr;
};

/**
 * Below diagram shows GRU pattern to fuse.
 * - the below pattern will be replaced with one GRU
 *  Main Graph:
 *           [In]    [CircleConst]    [CircleConst]    [CircleConst]    [CircleConst]
 *            |             |              |                |                 |
 *            V             |              |                |                 |
 *         [CircleWhile]<-----------------------------------------------------
 *            |
 *            V
 *      [CircleWhileOut]
 *            |
 *            V
 *          [Out]
 *
 * Condition Graph:
 *          [In]   [CircleConst] (scalar int32 value)
 *           |        |
 *           V        |
 *         [Less]------
 *           |
 *           V
 *         [Out]
 *
 *  Body Graph must contain:
 *  - 2 CircleFullyConnected nodes;
 *  - 3 CircleMul nodes;
 *  - 2 CircleLogistic nodes;
 *  - 2 CircleSplit nodes;
 *  - 6 CircleAdd nodes;
 *  - 1 CircleGather node;
 *  - 1 CircleReshape node;
 *  - 1 CircleSub node;
 *  - 1 CircleTanh node;
 *  - 6 CircleSplitOut nodes;
 *  - 5 CircleInput nodes;
 *  - 5 CircleOutput nodes;
 */
class GRUPattern1 final : public GRUPatternBase
{
public:
  GRUPattern1(luci::CircleWhileOut *candidate) : GRUPatternBase(candidate)
  {
    assert(candidate);
    _while_out_node = candidate;
  }

public:
  bool matched() override;
};

bool GRUPattern1::matched()
{
  // 0 - check while node
  _while_node = dynamic_cast<luci::CircleWhile *>(_while_out_node->input());
  if (_while_node == nullptr)
    return false;

  // 1 - check condition graph: only one Less operation
  // with scalar int const value
  {
    const auto cond_graph = _while_node->cond_graph();

    const auto cond_nodes = loco::active_nodes(loco::output_nodes(cond_graph));
    if (cond_nodes.size() != 4)
      return false;
    luci::CircleLess *less_node = nullptr;
    for (auto node : cond_nodes)
    {
      less_node = dynamic_cast<luci::CircleLess *>(node);
      if (less_node != nullptr)
        break;
    }

    // doesn't find Less node
    if (less_node == nullptr)
      return false;

    luci::CircleNode *less_input;
    if (not luci::fill(&less_input, &_less_const).with_commutative_args_of(less_node))
      return false;

    if (_less_const->dtype() != loco::DataType::S32)
      return false;

    if (_less_const->size<loco::DataType::S32>() != 1)
      return false;

    assert(_less_const->at<loco::DataType::S32>(0) > 0);
  }

  // 2 - Check while's input nodes
  // Save hidden state input node
  {
    if (_while_node->input_count() != 5)
      return false;

    // Save input node
    _ifm = dynamic_cast<luci::CircleNode *>(_while_node->input(4));
    if (_ifm == nullptr)
      return false;

    _hidden_input = dynamic_cast<luci::CircleConst *>(_while_node->input(3));
    if (_hidden_input == nullptr)
      return false;
  }

  // 3 - check body graph
  // TODO: add more checks
  {
    const auto body_graph = _while_node->body_graph();

    if (loco::input_nodes(body_graph).size() != 5)
      return false;

    if (loco::output_nodes(body_graph).size() != 5)
      return false;

    const auto body_nodes = loco::active_nodes(loco::output_nodes(body_graph));

    int32_t num_fc = 0;
    int32_t num_split = 0;
    int32_t num_logistic = 0;
    int32_t num_mul = 0;
    int32_t num_add = 0;
    int32_t num_sub = 0;
    int32_t num_reshape = 0;
    int32_t num_gather = 0;
    int32_t num_tanh = 0;
    int32_t num_split_out = 0;

    for (auto node : body_nodes)
    {
      auto circle_node = dynamic_cast<luci::CircleNode *>(node);
      switch (circle_node->opcode())
      {
        case luci::CircleOpcode::CIRCLECONST:
        case luci::CircleOpcode::CIRCLEINPUT:
        case luci::CircleOpcode::CIRCLEOUTPUT:
        case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
          break;
        case luci::CircleOpcode::FULLY_CONNECTED:
          num_fc++;
          break;
        case luci::CircleOpcode::SPLIT:
          num_split++;
          break;
        case luci::CircleOpcode::LOGISTIC:
          num_logistic++;
          break;
        case luci::CircleOpcode::MUL:
          num_mul++;
          break;
        case luci::CircleOpcode::ADD:
          num_add++;
          break;
        case luci::CircleOpcode::SUB:
          num_sub++;
          break;
        case luci::CircleOpcode::RESHAPE:
          num_reshape++;
          break;
        case luci::CircleOpcode::GATHER:
          num_gather++;
          break;
        case luci::CircleOpcode::TANH:
          num_tanh++;
          break;
        case luci::CircleOpcode::CIRCLESPLITOUT:
          num_split_out++;
          break;
        default:
          return false;
      }
    }

    if (num_fc != 2 or num_mul != 3 or num_logistic != 2 or num_split != 2 or num_add != 6 or
        num_gather != 1 or num_reshape != 1 or num_sub != 1 or num_tanh != 1 or num_split_out != 6)
      return false;

    // Find input and hidden FC weights and biases
    for (auto node : body_nodes)
    {
      auto *fc_node = dynamic_cast<luci::CircleFullyConnected *>(node);
      if (fc_node == nullptr)
        continue;

      const auto input_node = dynamic_cast<luci::CircleNode *>(fc_node->input());
      if (input_node == nullptr)
        return false;

      // Hidden FullyConnected input node  - is CircleInput node
      if (dynamic_cast<luci::CircleInput *>(input_node) != nullptr)
      {
        _weight_ih = dynamic_cast<luci::CircleConst *>(fc_node->weights());
        _bias_ih = dynamic_cast<luci::CircleConst *>(fc_node->bias());
      }
      // Input FullyConnected input node  - is CircleGather node
      else if (dynamic_cast<luci::CircleGather *>(input_node) != nullptr)
      {
        _weight_hh = dynamic_cast<luci::CircleConst *>(fc_node->weights());
        _bias_hh = dynamic_cast<luci::CircleConst *>(fc_node->bias());
      }
      else
      {
        return false;
      }
    }

    if (_weight_ih == nullptr or _weight_hh == nullptr)
      return false;
  }

  return true;
}

class FuseGRU final
{
public:
  FuseGRU(const GRUPatternBase *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleCustomOut *create_custom_gru(loco::Graph *graph);

private:
  const GRUPatternBase *_p;
};

template <loco::DataType T>
void copy_values(const luci::CircleConst *node, luci::CircleConst *cloned)
{
  assert(T == node->dtype());
  assert(T == cloned->dtype());

  const auto size = node->size<T>();
  cloned->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    cloned->at<T>(i) = node->at<T>(i);
}

luci::CircleConst *clone_circleconst(luci::CircleConst *node, loco::Graph *graph)
{
  auto cloned = graph->nodes()->create<luci::CircleConst>();

  if (cloned != nullptr)
  {
    // dtype/shape
    cloned->dtype(node->dtype());
    cloned->rank(node->rank());

    // values
    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
        copy_values<loco::DataType::FLOAT32>(node, cloned);
        break;

      case loco::DataType::U8:
        copy_values<loco::DataType::U8>(node, cloned);
        break;

      case loco::DataType::S8:
        copy_values<loco::DataType::S8>(node, cloned);
        break;

      case loco::DataType::S16:
        copy_values<loco::DataType::S16>(node, cloned);
        break;

      case loco::DataType::S32:
        copy_values<loco::DataType::S32>(node, cloned);
        break;

      case loco::DataType::S64:
        copy_values<loco::DataType::S64>(node, cloned);
        break;

      case loco::DataType::BOOL:
        copy_values<loco::DataType::BOOL>(node, cloned);
        break;

      default:
        assert(false);
    }
  }

  return cloned;
}

luci::CircleCustomOut *FuseGRU::create_custom_gru(loco::Graph *graph)
{
  assert(graph);

  auto weight_ih_cloned = clone_circleconst(_p->_weight_ih, graph);
  luci::copy_common_attributes(_p->_weight_ih, weight_ih_cloned);

  auto weight_hh_cloned = clone_circleconst(_p->_weight_hh, graph);
  luci::copy_common_attributes(_p->_weight_hh, weight_hh_cloned);

  luci::CircleNode *bias_ih_cloned = nullptr;
  if (_p->_bias_ih != nullptr)
  {
    bias_ih_cloned = clone_circleconst(_p->_bias_ih, graph);
    luci::copy_common_attributes(_p->_bias_ih, bias_ih_cloned);
  }
  else
  {
    bias_ih_cloned = _p->_pattern_last_node->graph()->nodes()->create<luci::CircleOutputExclude>();
  }

  luci::CircleNode *bias_hh_cloned = nullptr;
  if (_p->_bias_hh != nullptr)
  {
    bias_hh_cloned = clone_circleconst(_p->_bias_hh, graph);
    luci::copy_common_attributes(_p->_bias_hh, bias_hh_cloned);
  }
  else
  {
    bias_hh_cloned = _p->_pattern_last_node->graph()->nodes()->create<luci::CircleOutputExclude>();
  }

  auto hidden_input_cloned = clone_circleconst(_p->_hidden_input, graph);
  luci::copy_common_attributes(_p->_hidden_input, hidden_input_cloned);

  auto less_const_cloned = clone_circleconst(_p->_less_const, graph);
  luci::copy_common_attributes(_p->_less_const, less_const_cloned);

  // Create and configure new CircleCustom operation.
  auto fused_gru = _p->_while_node->graph()->nodes()->create<luci::CircleCustom>(7, 1);
  auto custom_out = _p->_while_node->graph()->nodes()->create<luci::CircleCustomOut>();

  fused_gru->custom_code("CircleGru");
  fused_gru->inputs(0, _p->_ifm);
  fused_gru->inputs(1, weight_ih_cloned);
  fused_gru->inputs(2, weight_hh_cloned);
  fused_gru->inputs(3, bias_ih_cloned);
  fused_gru->inputs(4, bias_hh_cloned);
  fused_gru->inputs(5, hidden_input_cloned);
  fused_gru->inputs(6, less_const_cloned);

  fused_gru->name("CircleGru");

  fused_gru->shape_status(luci::ShapeStatus::VALID);

  fused_gru->rank(_p->_while_out_node->rank());
  for (uint32_t i = 0; i < fused_gru->rank(); ++i)
  {
    fused_gru->dim(i).set(_p->_while_out_node->dim(i).value());
  }

  fused_gru->dtype(loco::DataType::FLOAT32);

  custom_out->input(fused_gru);
  custom_out->name("CircleGruOut");
  custom_out->rank(_p->_while_out_node->rank());
  for (uint32_t i = 0; i < fused_gru->rank(); ++i)
  {
    custom_out->dim(i).set(fused_gru->dim(i).value());
  }
  custom_out->dtype(loco::DataType::FLOAT32);
  custom_out->shape_status(luci::ShapeStatus::VALID);
  custom_out->index(0);

  return custom_out;
}

void FuseGRU::apply()
{
  auto graph = _p->_pattern_last_node->graph();

  auto gru_out = create_custom_gru(graph);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_while_node), luci::get_origin(_p->_while_out_node),
    luci::get_origin(_p->_weight_hh), luci::get_origin(_p->_weight_ih)};

  luci::add_origin(gru_out, luci::composite_origin(origin_vec));

  replace(_p->_pattern_last_node).with(gru_out);
}

} // namespace

namespace
{

bool fuse_gru(luci::CircleWhileOut *while_out_node)
{
  assert(while_out_node);

  // check first pattern
  GRUPattern1 pattern(while_out_node);
  if (pattern.matched())
  {
    FuseGRU fuse(&pattern);
    fuse.apply();
    return true;
  }

  return false;
}

} // namespace

namespace luci
{

bool FuseGRUPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto while_out_node = dynamic_cast<luci::CircleWhileOut *>(node);
    if (not while_out_node)
      continue;

    if (fuse_gru(while_out_node))
      changed = true;
  }

  return changed;
}

} // namespace luci
