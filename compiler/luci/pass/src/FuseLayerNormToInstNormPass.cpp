/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseLayerNormToInstNormPass.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>

#include <cassert>

/**
 * Below diagram shows decomposed LayerNorm pattern to fuse to LayerNorm
 *
 *          input
 *            |
 *     +------+
 *     |      |
 *     |      V
 *     |   mean_in
 *     |      |
 *     |      V
 *     +---> sub
 *            |
 *     +------+
 *     |      |
 *     |      V
 *     |     mul
 *     |      |
 *     |      V
 *     |   mean_mul
 *     |      |
 *     |      V
 *     |    add_eps
 *     |      |
 *     |      V
 *     |    rsqrt
 *     |      |
 *     |      V
 *     +-> mul_sub
 *            |
 *            V
 *          output
 *
 * Below diagram shows decomposed graph with InstanceNorm from LayerNorm
 *
 *          input
 *         (N,L,D)
 *            |
 *            V
 *        transpose_in
 *         (N,D,L)
 *            |
 *            V
 *        reshape_in
 *        (N,1,D,L)
 *            |
 *            V
 *       instancenorm
 *        (N,1,D,L)
 *            |
 *            V
 *        reshape_out
 *         (N,D,L)
 *            |
 *            V
 *       transpose_out
 *          (N,L,D)
 *            |
 *            V
 *          output
 */

namespace luci
{

namespace
{

class LayerNormPattern
{
public:
  LayerNormPattern(luci::CircleMul *candidate) { _mul_sub = candidate; }
  ~LayerNormPattern() = default;

public:
  bool matched(void);

public:
  uint32_t _batch = 0;
  uint32_t _length = 0;
  uint32_t _dim = 0;
  luci::CircleNode *_input = nullptr;
  luci::CircleMean *_mean_in = nullptr;
  luci::CircleSub *_sub = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleMean *_mean_mul = nullptr;
  luci::CircleAdd *_add_eps = nullptr;
  luci::CircleRsqrt *_rsqrt = nullptr;
  luci::CircleMul *_mul_sub = nullptr;
  float _epsilon = 0.00001f;
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool LayerNormPattern::matched(void)
{
  CHECK_OR_FALSE(_mul_sub != nullptr);
  CHECK_OR_FALSE(_mul_sub->rank() == 3);
  CHECK_OR_FALSE(_mul_sub->dtype() == loco::DataType::FLOAT32);

  CHECK_OR_FALSE(luci::fill(&_sub, &_rsqrt).with_commutative_args_of(_mul_sub));
  _add_eps = dynamic_cast<luci::CircleAdd *>(_rsqrt->x());
  CHECK_OR_FALSE(_add_eps != nullptr);

  luci::CircleConst *add_epsilon = nullptr;
  CHECK_OR_FALSE(luci::fill(&_mean_mul, &add_epsilon).with_commutative_args_of(_add_eps));
  CHECK_OR_FALSE(_mean_mul->keep_dims());
  CHECK_OR_FALSE(add_epsilon->dtype() == loco::DataType::FLOAT32);
  CHECK_OR_FALSE(add_epsilon->size<loco::DataType::FLOAT32>() == 1);

  _mul = dynamic_cast<luci::CircleMul *>(_mean_mul->input());
  CHECK_OR_FALSE(_mul != nullptr);
  luci::CircleConst *mean_mul_indices =
    dynamic_cast<luci::CircleConst *>(_mean_mul->reduction_indices());
  CHECK_OR_FALSE(mean_mul_indices != nullptr);
  // TODO check mean_mul_indices value

  luci::CircleSub *sub1 = nullptr;
  luci::CircleSub *sub2 = nullptr;
  CHECK_OR_FALSE(luci::fill(&sub1, &sub2).with_commutative_args_of(_mul));
  CHECK_OR_FALSE(sub1 == _sub);
  CHECK_OR_FALSE(sub2 == _sub);

  _input = dynamic_cast<luci::CircleNode *>(_sub->x());
  CHECK_OR_FALSE(_input != nullptr);
  _mean_in = dynamic_cast<luci::CircleMean *>(_sub->y());
  CHECK_OR_FALSE(_mean_in != nullptr);
  CHECK_OR_FALSE(_mean_in->keep_dims());

  luci::CircleNode *input = dynamic_cast<luci::CircleNode *>(_mean_in->input());
  CHECK_OR_FALSE(input == _input);
  luci::CircleConst *mean_in_indices =
    dynamic_cast<luci::CircleConst *>(_mean_in->reduction_indices());
  CHECK_OR_FALSE(mean_in_indices != nullptr);
  // TODO check mean_in_indices value

  return true;
}

} // namespace

namespace
{

class FuseLayerNorm final
{
public:
  FuseLayerNorm(const LayerNormPattern *p) : _p(p) {}

public:
  void apply(void);

private:
  luci::CircleLayerNorm *create_layernorm(loco::Graph *g);

private:
  const LayerNormPattern *_p;
};

luci::CircleConst *make_const(loco::Graph *g, uint32_t dim, float value)
{
  auto const_one = g->nodes()->create<luci::CircleConst>();
  const_one->dtype(loco::DataType::FLOAT32);
  const_one->rank(1);
  const_one->size<loco::DataType::FLOAT32>(dim);
  for (uint32_t d = 0; d < dim; ++d)
    const_one->at<loco::DataType::FLOAT32>(d) = value;
  return const_one;
}

luci::CircleLayerNorm *FuseLayerNorm::create_layernorm(loco::Graph *g)
{
  assert(g);

  CircleConst *gamma = make_const(g, 1, 1.0f);
  CircleConst *beta = make_const(g, 1, 0.0f);

  auto ln = g->nodes()->create<luci::CircleLayerNorm>();
  ln->input(_p->_input);
  ln->gamma(gamma);
  ln->beta(beta);
  ln->epsilon(_p->_epsilon);
  ln->name(_p->_mul_sub->name() + "_layernorm");

  gamma->name(_p->_mul_sub->name() + "_layernorm/gamma");
  beta->name(_p->_mul_sub->name() + "_layernorm/beta");

  return ln;
}

void FuseLayerNorm::apply()
{
  auto g = _p->_mul_sub->graph();
  auto layernorm = create_layernorm(g);

  // set origin
  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> origin_vec{
    luci::get_origin(_p->_mean_in),  luci::get_origin(_p->_sub),     luci::get_origin(_p->_mul),
    luci::get_origin(_p->_mean_mul), luci::get_origin(_p->_add_eps), luci::get_origin(_p->_rsqrt)};

  luci::add_origin(layernorm, luci::composite_origin(origin_vec));

  replace(_p->_mul_sub).with(layernorm);
}

} // namespace

namespace
{

class DecomposeToInstanceNorm final
{
public:
  DecomposeToInstanceNorm(CircleLayerNorm *node) : _layernorm(node) {}

public:
  void apply(void);

private:
  luci::CircleTranspose *create_subgraph(loco::Graph *g);

private:
  CircleLayerNorm *_layernorm;
};

luci::CircleConst *create_transpose_perm(loco::Graph *g, const std::initializer_list<uint32_t> perm)
{
  auto const_perm = g->nodes()->create<luci::CircleConst>();
  const_perm->dtype(loco::DataType::S32);
  const_perm->size<loco::DataType::S32>(perm.size());
  const_perm->rank(1);
  const_perm->dim(0) = perm.size();
  uint32_t i = 0;
  for (auto p = perm.begin(); p != perm.end(); ++p, ++i)
    const_perm->at<loco::DataType::S32>(i) = *p;
  const_perm->shape_status(luci::ShapeStatus::VALID);
  return const_perm;
}

void setNewShape(luci::CircleReshape *reshape, const std::initializer_list<uint32_t> shape)
{
  reshape->newShape()->rank(shape.size());
  uint32_t i = 0;
  for (auto s = shape.begin(); s != shape.end(); ++s, ++i)
  {
    reshape->newShape()->dim(i) = *s;
  }
}

// creates with
//   Transpose [N, D, L] -> Reshape [N, 1, D, L] ->
//   InstanceNorm [N, 1, D, L] ->
//   Reshape [N, D, L] -> Transpose [N, L, D]
luci::CircleTranspose *DecomposeToInstanceNorm::create_subgraph(loco::Graph *g)
{
  auto input = loco::must_cast<luci::CircleNode *>(_layernorm->input());

  auto name = _layernorm->name();
  auto origin = luci::get_origin(_layernorm);
  assert(_layernorm->rank() == 3);
  auto dim_N = _layernorm->dim(0).value();
  auto dim_L = _layernorm->dim(1).value();
  auto dim_D = _layernorm->dim(2).value();

  auto perm_in = create_transpose_perm(g, {0, 2, 1});
  perm_in->name(name + "/Transpose1/perm");
  luci::add_origin(perm_in, origin);
  auto transpose_in = g->nodes()->create<luci::CircleTranspose>();
  transpose_in->a(input);
  transpose_in->perm(perm_in);
  transpose_in->name(name + "/Transpose1");
  luci::add_origin(transpose_in, origin);

  auto reshape_in_d = g->nodes()->create<luci::CircleOutputDummy>();
  reshape_in_d->name(name + "/Reshape1/dummy");
  reshape_in_d->dtype(loco::DataType::S32);
  reshape_in_d->rank(0);
  auto rehape_in = g->nodes()->create<luci::CircleReshape>();
  rehape_in->tensor(transpose_in);
  rehape_in->shape(reshape_in_d);
  setNewShape(rehape_in, {dim_N, 1, dim_D, dim_L});
  rehape_in->name(name + "/Reshape1");
  luci::add_origin(rehape_in, origin);

  auto *instnorm = g->nodes()->create<luci::CircleInstanceNorm>();
  instnorm->input(rehape_in);
  instnorm->gamma((_layernorm->gamma()));
  instnorm->beta((_layernorm->beta()));
  instnorm->fusedActivationFunction(luci::FusedActFunc::NONE);
  instnorm->epsilon(_layernorm->epsilon());
  instnorm->name(name + "/InstanceNorm");
  luci::add_origin(instnorm, origin);

  auto reshape_out_d = g->nodes()->create<luci::CircleOutputDummy>();
  reshape_out_d->name(name + "/Reshape2/dummy");
  reshape_out_d->dtype(loco::DataType::S32);
  reshape_out_d->rank(0);
  auto rehape_out = g->nodes()->create<luci::CircleReshape>();
  rehape_out->tensor(instnorm);
  rehape_out->shape(reshape_out_d);
  setNewShape(rehape_out, {dim_N, dim_D, dim_L});
  rehape_out->name(name + "/Reshape2");
  luci::add_origin(rehape_out, origin);

  auto perm_out = create_transpose_perm(g, {0, 2, 1});
  perm_out->name(name + "/Transpose2/perm");
  luci::add_origin(perm_out, origin);
  auto transpose_out = g->nodes()->create<luci::CircleTranspose>();
  transpose_out->a(rehape_out);
  transpose_out->perm(perm_out);
  transpose_out->name(name + "/Transpose2");
  luci::add_origin(transpose_out, origin);

  return transpose_out;
}

void DecomposeToInstanceNorm::apply(void)
{
  auto g = _layernorm->graph();
  auto transpose = create_subgraph(g);

  replace(_layernorm).with(transpose);
}

} // namespace

namespace
{

bool fuse_layernorm(luci::CircleMul *mul)
{
  assert(mul);

  LayerNormPattern pattern(mul);
  if (pattern.matched())
  {
    FuseLayerNorm fuse(&pattern);
    fuse.apply();
    return true;
  }
  return false;
}

bool convert_to_instancenorm(luci::CircleLayerNorm *layernorm)
{
  CHECK_OR_FALSE(layernorm->rank() == 3);
  CHECK_OR_FALSE(layernorm->dtype() == loco::DataType::FLOAT32);

  DecomposeToInstanceNorm decomp(layernorm);
  decomp.apply();

  return true;
}

} // namespace

bool FuseLayerNormToInstNormPass::run(loco::Graph *g)
{
  bool changed = false;

  // fuse certain sub-graph to CircleLayerNorm
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mul = dynamic_cast<luci::CircleMul *>(node);
    if (mul != nullptr)
    {
      if (fuse_layernorm(mul))
        changed = true;
    }
  }
  // if there is any conversion, return as changed so that shape-dtype is infered
  if (changed)
    return changed;

  // convert CircleLayerNorm to certain sub-graph with CircleInstanceNorm
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto layernorm = dynamic_cast<luci::CircleLayerNorm *>(node);
    if (layernorm != nullptr)
    {
      if (convert_to_instancenorm(layernorm))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
