/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/CanonicalizePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco/IR/DataType.h>

#include <limits>

#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

namespace
{

/**
 * Convert S64 CircleConst paddings to S32
 */
template <class PAD> bool paddings_to_s32(PAD *pad)
{
  // check conditions
  auto paddings = dynamic_cast<luci::CircleConst *>(pad->paddings());
  CHECK_OR_FALSE(paddings);
  CHECK_OR_FALSE(paddings->dtype() == loco::DataType::S64);

  // TODO relocate to helpers/CreateCircleConst.h when necessary
  auto num_elements = paddings->size<loco::DataType::S64>();
  auto hval = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  auto lval = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
  for (uint32_t i = 0; i < num_elements; i++)
  {
    auto v64 = paddings->at<loco::DataType::S64>(i);
    CHECK_OR_FALSE(v64 <= hval);
    CHECK_OR_FALSE(v64 >= lval);
  }

  auto paddings_s32 = pad->graph()->nodes()->template create<luci::CircleConst>();
  paddings_s32->name(paddings->name() + "_S32");
  paddings_s32->dtype(loco::DataType::S32);
  paddings_s32->rank(paddings->rank());
  for (uint32_t i = 0; i < paddings->rank(); i++)
    paddings_s32->dim(i).set(paddings->dim(i).value());
  paddings_s32->shape_status(luci::ShapeStatus::VALID);
  luci::add_origin(paddings_s32, luci::get_origin(paddings));

  paddings_s32->template size<loco::DataType::S32>(num_elements);
  for (uint32_t i = 0; i < num_elements; i++)
  {
    auto v64 = paddings->at<loco::DataType::S64>(i);
    paddings_s32->template at<loco::DataType::S32>(i) = static_cast<int32_t>(v64);
  }

  // replace paddings with S32 dtype
  pad->paddings(paddings_s32);

  return true;
}

} // namespace

namespace luci
{

/**
 * Canonicalize circle nodes
 */
bool CanonicalizePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto pad = dynamic_cast<luci::CirclePad *>(node))
    {
      if (paddings_to_s32(pad))
        changed = true;
    }
    else if (auto padv2 = dynamic_cast<luci::CirclePadV2 *>(node))
    {
      if (paddings_to_s32(padv2))
        changed = true;
    }

    // TODO add more canonicalization
  }

  return changed;
}

} // namespace luci
