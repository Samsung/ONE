/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveDuplicateConstPass.h"

#include <luci/Log.h>

namespace
{

bool compare_quant_params(luci::CircleConst *left, luci::CircleConst *right)
{
  const auto left_quant_param = left->quantparam();
  const auto right_quant_param = right->quantparam();

  if (left_quant_param == right_quant_param)
    return true;

  if (left_quant_param != nullptr and right_quant_param != nullptr)
  {
    if (left_quant_param->scale == right_quant_param->scale and
        left_quant_param->quantized_dimension == right_quant_param->quantized_dimension and
        left_quant_param->zerop == right_quant_param->zerop and
        left_quant_param->min == right_quant_param->min and
        left_quant_param->max == right_quant_param->max)
    {
      return true;
    }
  }
  return false;
}

bool compare_dim_values(luci::CircleConst *left, luci::CircleConst *right)
{
  const auto left_rank = left->rank();
  const auto right_rank = right->rank();

  if (left_rank != right_rank)
    return false;

  for (uint32_t i = 0; i < left_rank; ++i)
  {
    if (left->dim(i).value() != right->dim(i).value())
      return false;
  }

  return true;
}

template <loco::DataType DT> bool is_equal_consts(luci::CircleConst *left, luci::CircleConst *right)
{
  if (not compare_quant_params(left, right))
    return false;

  if (not compare_dim_values(left, right))
    return false;

  for (uint32_t i = 0; i < left->size<DT>(); ++i)
  {
    if (left->at<DT>(i) != right->at<DT>(i))
      return false;
  }

  return true;
}

} // namespace

namespace luci
{

bool RemoveDuplicateConstPass::remove_duplicate_const()
{
  bool changed = false;

  for (auto &cur_pair : _sum_to_const)
  {
    // if single const - continue
    if (cur_pair.second.size() == 1)
      continue;

    for (auto reference_const : cur_pair.second)
    {
      if (reference_const == nullptr)
        continue;

      for (uint32_t i = 0; i < cur_pair.second.size(); ++i)
      {
        auto cur_const = cur_pair.second.at(i);
        if (cur_const == nullptr or cur_const == reference_const)
          continue;

        if (cur_const->dtype() != reference_const->dtype())
          continue;

        bool is_equal = false;

        switch (cur_const->dtype())
        {
          case loco::DataType::FLOAT32:
            is_equal = is_equal_consts<loco::DataType::FLOAT32>(reference_const, cur_const);
            break;
          case loco::DataType::S32:
            is_equal = is_equal_consts<loco::DataType::S32>(reference_const, cur_const);
            break;
          case loco::DataType::S16:
            is_equal = is_equal_consts<loco::DataType::S16>(reference_const, cur_const);
            break;
          case loco::DataType::S8:
            is_equal = is_equal_consts<loco::DataType::S8>(reference_const, cur_const);
            break;
          case loco::DataType::S4:
            is_equal = is_equal_consts<loco::DataType::S4>(reference_const, cur_const);
            break;
          case loco::DataType::U8:
            is_equal = is_equal_consts<loco::DataType::U8>(reference_const, cur_const);
            break;
          case loco::DataType::U4:
            is_equal = is_equal_consts<loco::DataType::U4>(reference_const, cur_const);
            break;
          default:
            continue;
        }

        if (not is_equal)
          continue;

        loco::replace(cur_const).with(reference_const);

        // Remove from next checking
        cur_pair.second[i] = nullptr;

        changed = true;
      }
    }
  }

  return changed;
}

template <loco::DataType DT>
void RemoveDuplicateConstPass::add_to_map(luci::CircleConst *const_node)
{
  const auto const_size = const_node->size<DT>();
  float sum = 0.0;

  for (uint32_t i = 0; i < const_size; ++i)
  {
    sum += const_node->at<DT>(i);
  }

  if (_sum_to_const.find(sum) == _sum_to_const.end())
  {
    _sum_to_const[sum] = {const_node};
  }
  else
  {
    _sum_to_const.at(sum).push_back(const_node);
  }
}

/**
 * Remove duplicate Const nodes.
 *
 * BEFORE
 *    [CircleNode]   [CircleConst]
 *          |        /
 *          |      /
 *    [CircleNode]    [CircleConst]
 *          |        /
 *          |      /
 *    [CircleNode]
 *
 * AFTER
 *
 *    [CircleNode]   [CircleConst]
 *          |        /     /
 *          |      /     /
 *    [CircleNode]     /
 *          |        /
 *          |      /
 *    [CircleNode]
 *
 */
bool RemoveDuplicateConstPass::run(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (const_node == nullptr)
      continue;

    switch (const_node->dtype())
    {
      case loco::DataType::FLOAT32:
        add_to_map<loco::DataType::FLOAT32>(const_node);
        break;
      case loco::DataType::S32:
        add_to_map<loco::DataType::S32>(const_node);
        break;
      case loco::DataType::S16:
        add_to_map<loco::DataType::S16>(const_node);
        break;
      case loco::DataType::S8:
        add_to_map<loco::DataType::S8>(const_node);
        break;
      case loco::DataType::S4:
        add_to_map<loco::DataType::S4>(const_node);
        break;
      case loco::DataType::U8:
        add_to_map<loco::DataType::U8>(const_node);
        break;
      case loco::DataType::U4:
        add_to_map<loco::DataType::U4>(const_node);
        break;
      default:
        continue;
    }
  }

  return remove_duplicate_const();
}

} // namespace luci
