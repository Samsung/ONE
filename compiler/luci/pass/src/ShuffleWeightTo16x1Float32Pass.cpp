/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ShuffleWeightTo16x1Float32Pass.h"

#include <luci/IR/CircleNodes.h>

#include <cassert>
#include <vector>

namespace
{

bool satisfy_precondition(luci::CircleFullyConnected *fc)
{
  // check if it's already been shuffled
  if (fc->weights_format() != luci::CircleFullyConnected::WeightsFormat::DEFAULT)
    return false;

  // check if its data type is FLOAT32
  if (fc->dtype() != loco::DataType::FLOAT32)
    return false;

  auto weights = loco::must_cast<luci::CircleConst *>(fc->weights());
  // rank must be 2
  if (weights->rank() != 2)
    return false;

  // check if it has sparsity parameter
  if (weights->sparsityparam())
    return false;

  // check if the number of row of FullyConnected's weight is a multiple of 16
  const uint32_t MULTIPLE = 16;
  uint32_t rows = weights->dim(0).value();
  if (rows % MULTIPLE)
    return false;

  return true;
}

// get FullyConnected op vector that has same tensor
void get_FCs_having_same_tensor(std::vector<luci::CircleFullyConnected *> &fc_vec, loco::Graph *g,
                                luci::CircleFullyConnected *fc)
{
  auto the_tensor = fc->weights();
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (fc->weights() == the_tensor)
      fc_vec.push_back(fc);
  }
}

luci::CircleConst *shuffle_weight(luci::CircleFullyConnected *fc)
{
  auto the_weights = loco::must_cast<luci::CircleConst *>(fc->weights());

  // create CircleConst where shuffled data will be stored
  luci::CircleConst *new_weights = fc->graph()->nodes()->create<luci::CircleConst>();
  new_weights->dtype(loco::DataType::FLOAT32);
  new_weights->size<loco::DataType::FLOAT32>(the_weights->size<loco::DataType::FLOAT32>());
  new_weights->rank(the_weights->rank());
  new_weights->shape_status(the_weights->shape_status());
  for (uint32_t r = 0; r < new_weights->rank(); r++)
  {
    new_weights->dim(r).set(the_weights->dim(r).value());
  }

  // suffle weight
  const uint32_t MULTIPLE = 16;
  const uint32_t rows = the_weights->dim(0).value();
  const uint32_t cols = the_weights->dim(1).value();
  const uint32_t r_step = rows / MULTIPLE;
  uint32_t index = 0;
  for (uint32_t r = 0; r < r_step; r++)
  {
    for (uint32_t c = 0; c < cols; c++)
    {
      for (uint32_t i = 0; i < MULTIPLE; i++)
      {
        new_weights->at<loco::DataType::FLOAT32>(index++) =
          the_weights->at<loco::DataType::FLOAT32>((r * MULTIPLE + i) * cols + c);
      }
    }
  }

  return new_weights;
}

} // namespace

namespace luci
{

bool ShuffleWeightTo16x1Float32Pass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (not satisfy_precondition(fc))
      continue;

    std::vector<luci::CircleFullyConnected *> fc_vec;
    get_FCs_having_same_tensor(fc_vec, g, fc);
    auto new_weights = shuffle_weight(fc);

    // replace to new weights
    for (const auto fc : fc_vec)
    {
      fc->weights(new_weights);
      fc->weights_format(luci::CircleFullyConnected::WeightsFormat::SHUFFLED16x1FLOAT32);
    }

    changed = true;
  }

  return changed;
}

} // namespace luci
