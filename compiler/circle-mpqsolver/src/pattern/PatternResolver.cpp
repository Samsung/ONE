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

#include "PatternResolver.h"

#include <map>
#include <set>
#include <vector>

using namespace mpqsolver::pattern;

using LayerParam = luci::CircleQuantizer::Options::LayerParam;

namespace
{

/**
 * SUBGRAPH PATTERN
 *  LayerNorm := (ifm - |ifm|) / sqrt(|(ifm - |ifm|)^2| + eps)
 *    - |x|: mean of x
 *    - Below diagram shows LayerNorm pattern to quantize.
 *
 *                 [In]
 *                   |
 *                   V
 *                  ifm -----+   (reduction indicies)
 *                   |       |       |
 *                   |       V       V
 *                   |      mean_of_ifm
 *                   V       |
 *     +------------sub <----+
 *     |             |
 *     |             V           (reduction indicies)
 *     |          sub_squared           |
 *     |             |                  |
 *     |             V                  |
 *     |       mean_as_variance <-------+    (const_as_eps)
 *     |             |                          |
 *     |             V                          |
 *     |          add_eps<----------------------+
 *     |             |
 *     |             V
 *     |           rsqrt
 *     |             |
 *     |             V
 *     +------>mul_as_terminal
 *                   |
 *                   V
 *                 [Out]
 *
 */
class LayerNormPattern final
{
public:
  LayerNormPattern(luci::CircleMul *candidate)
  {
    assert(candidate);
    mul_as_terminal = candidate;
  }

public:
  bool matched();

  std::vector<luci::CircleNode *> get_q16_nodes()
  {
    return {sub_squared, mean_as_variance, add_eps, rsqrt};
  }

  std::vector<luci::CircleNode *> get_q8_nodes() { return {mean_of_ifm, sub, mul_as_terminal}; }

public:
  loco::Node *ifm = nullptr;
  luci::CircleMean *mean_of_ifm = nullptr;      // = |ifm|
  luci::CircleSub *sub = nullptr;               // = ifm - |ifm|
  luci::CircleMul *sub_squared = nullptr;       // = (ifm - |ifm|)^2
  luci::CircleMean *mean_as_variance = nullptr; // = |(ifm - |ifm|)^2|
  luci::CircleAdd *add_eps = nullptr;           // = |(ifm - |ifm|)^2| + eps
  luci::CircleRsqrt *rsqrt = nullptr;           // = 1.0 / sqrt(|(ifm - |ifm|)^2| + eps)
  luci::CircleMul *mul_as_terminal = nullptr;   // = (ifm - |ifm|) / sqrt(|(ifm - |ifm|)^2| + eps)
};

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

bool LayerNormPattern::matched()
{
  sub = dynamic_cast<luci::CircleSub *>(mul_as_terminal->x());
  rsqrt = dynamic_cast<luci::CircleRsqrt *>(mul_as_terminal->y());
  if (!sub || !rsqrt)
  {
    sub = dynamic_cast<luci::CircleSub *>(mul_as_terminal->y());
    rsqrt = dynamic_cast<luci::CircleRsqrt *>(mul_as_terminal->x());
  }
  CHECK_OR_FALSE(rsqrt != nullptr && sub != nullptr);

  ifm = dynamic_cast<luci::CircleNode *>(sub->x());
  mean_of_ifm = dynamic_cast<luci::CircleMean *>(sub->y());
  CHECK_OR_FALSE(ifm != nullptr && mean_of_ifm != nullptr);

  add_eps = dynamic_cast<luci::CircleAdd *>(rsqrt->x());
  CHECK_OR_FALSE(add_eps != nullptr);

  auto const *eps = dynamic_cast<luci::CircleConst *>(add_eps->x());
  mean_as_variance = dynamic_cast<luci::CircleMean *>(add_eps->y());
  if (!eps || !mean_as_variance)
  {
    eps = dynamic_cast<luci::CircleConst *>(add_eps->y());
    mean_as_variance = dynamic_cast<luci::CircleMean *>(add_eps->x());
  }

  CHECK_OR_FALSE(eps != nullptr && mean_as_variance != nullptr);

  // eps should be scalar value
  CHECK_OR_FALSE(eps->size<loco::DataType::FLOAT32>() == 1);

  sub_squared = dynamic_cast<luci::CircleMul *>(mean_as_variance->input());
  CHECK_OR_FALSE(sub_squared != nullptr);

  // check that sub_squared = sub * sub
  CHECK_OR_FALSE(sub_squared->x() == sub_squared->y() && sub_squared->x() == sub);

  auto const mean_as_variance_indices =
    dynamic_cast<luci::CircleConst *>(mean_as_variance->reduction_indices());
  auto const mean_of_ifm_indices =
    dynamic_cast<luci::CircleConst *>(mean_of_ifm->reduction_indices());

  // check validity of reduction indices
  CHECK_OR_FALSE(mean_of_ifm_indices != nullptr && mean_as_variance_indices != nullptr);

  // check dtype of reduction indices
  CHECK_OR_FALSE(mean_of_ifm_indices->dtype() == loco::DataType::S32 &&
                 mean_as_variance_indices->dtype() == loco::DataType::S32);

  // reduction indices of both mean operations must be the same
  CHECK_OR_FALSE(mean_as_variance_indices->size<loco::DataType::S32>() ==
                 mean_of_ifm_indices->size<loco::DataType::S32>());

  std::set<int32_t> set_of_mean_as_variance_indices;
  std::set<int32_t> set_of_mean_of_ifm_indices;
  for (uint32_t index = 0; index < mean_as_variance_indices->size<loco::DataType::S32>(); index++)
  {
    set_of_mean_as_variance_indices.insert(
      mean_as_variance_indices->at<loco::DataType::S32>(index));
    set_of_mean_of_ifm_indices.insert(mean_of_ifm_indices->at<loco::DataType::S32>(index));
  }
  // now make sure that reduction indices of mean_as_variance are the same as mean_of_ifm
  CHECK_OR_FALSE(set_of_mean_as_variance_indices == set_of_mean_of_ifm_indices);

  return true;
}

#undef CHECK_OR_FALSE

} // namespace

std::map<luci::CircleNode *, LayerParam>
Q8LayerNormWithQ16VarianceResolver::resolve(const luci::Module *module)
{
  if (!module)
  {
    throw std::runtime_error("no module for pattern resolving");
  }

  std::map<luci::CircleNode *, LayerParam> nodes_params;
  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    for (auto node : loco::active_nodes(loco::output_nodes(graph)))
    {
      auto const mul = dynamic_cast<luci::CircleMul *>(node);
      if (!mul)
        continue;

      LayerNormPattern pattern(mul);
      if (!pattern.matched())
        continue;

      // set quantization parameters of recognized pattern
      for (auto q16_node : pattern.get_q16_nodes())
      {
        LayerParam param = {q16_node->name(), "int16", "channel"};
        nodes_params[q16_node] = param;
      }

      for (auto q8_node : pattern.get_q8_nodes())
      {
        LayerParam param = {q8_node->name(), "uint8", "channel"};
        nodes_params[q8_node] = param;
      }
    }
  }

  return nodes_params;
}
