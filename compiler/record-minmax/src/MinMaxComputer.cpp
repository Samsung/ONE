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

#include "MinMaxComputer.h"
#include "RecordFunction.h"

#include <luci/IR/CircleQuantParam.h>

namespace record_minmax
{

void PercentileComputer::update_qparam(
  const std::unordered_map<const luci::CircleNode *, MinMaxVectors> *minmax_map)
{
  if (minmax_map == nullptr)
    throw std::invalid_argument("minmax_map is nullptr");

  for (auto iter = minmax_map->begin(); iter != minmax_map->end(); ++iter)
  {
    auto node = iter->first;
    auto minmax = iter->second;

    auto min = getNthPercentile(minmax.min_vector, _min_percentile);
    auto max = getNthPercentile(minmax.max_vector, _max_percentile);

    auto quantparam = std::make_unique<luci::CircleQuantParam>();
    quantparam->min.push_back(min);
    quantparam->max.push_back(max);

    assert(node->quantparam() == nullptr);

    auto mutable_node = const_cast<luci::CircleNode *>(node);
    mutable_node->quantparam(std::move(quantparam));
  }
}

void MovingAvgComputer::update_qparam(
  const std::unordered_map<const luci::CircleNode *, MinMaxVectors> *minmax_map)
{
  if (minmax_map == nullptr)
    throw std::invalid_argument("minmax_map is nullptr");

  for (auto iter = minmax_map->begin(); iter != minmax_map->end(); ++iter)
  {
    auto node = iter->first;
    auto minmax = iter->second;

    auto min = getMovingAverage(minmax.min_vector, 1 - _update_const, _batch_size, true);
    auto max = getMovingAverage(minmax.max_vector, 1 - _update_const, _batch_size, false);

    auto quantparam = std::make_unique<luci::CircleQuantParam>();
    quantparam->min.push_back(min);
    quantparam->max.push_back(max);

    assert(node->quantparam() == nullptr);

    auto mutable_node = const_cast<luci::CircleNode *>(node);
    mutable_node->quantparam(std::move(quantparam));
  }
}

std::unique_ptr<MinMaxComputer> make_percentile_computer(float min_percentile, float max_percentile)
{
  return std::make_unique<PercentileComputer>(min_percentile, max_percentile);
}

std::unique_ptr<MinMaxComputer> make_moving_avg_computer(uint32_t batch_size,
                                                         float moving_avg_const)
{
  return std::make_unique<MovingAvgComputer>(batch_size, moving_avg_const);
}

} // namespace record_minmax
