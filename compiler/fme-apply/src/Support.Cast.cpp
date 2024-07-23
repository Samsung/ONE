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

#include "Support.Cast.h"

namespace fme_apply
{

luci::CircleCustom *to_pre_scale(loco::Node *node)
{
  auto pre_scale = dynamic_cast<luci::CircleCustom *>(node);
  if (not pre_scale)
    return nullptr;

  if (pre_scale->custom_code() != "PreScale")
    return nullptr;

  // TODO Return false?
  assert(pre_scale->numInputs() == 2); // FIX_PreScale_UNLESS

  return pre_scale;
}

luci::CircleCustom *to_post_scale(loco::Node *node)
{
  auto post_scale = dynamic_cast<luci::CircleCustom *>(node);
  if (not post_scale)
    return nullptr;

  if (post_scale->custom_code() != "PostScale")
    return nullptr;

  // TODO Return false?
  assert(post_scale->numInputs() == 2); // FIX_PostScale_UNLESS

  return post_scale;
}

luci::CircleCustom *to_pre_shift(loco::Node *node)
{
  auto pre_shift = dynamic_cast<luci::CircleCustom *>(node);
  if (not pre_shift)
    return nullptr;

  if (pre_shift->custom_code() != "PreShift")
    return nullptr;

  // TODO Return false?
  assert(pre_shift->numInputs() == 2); // FIX_PreShift_UNLESS

  return pre_shift;
}

luci::CircleCustom *to_post_shift(loco::Node *node)
{
  auto post_shift = dynamic_cast<luci::CircleCustom *>(node);
  if (not post_shift)
    return nullptr;

  if (post_shift->custom_code() != "PostShift")
    return nullptr;

  // TODO Return false?
  assert(post_shift->numInputs() == 2); // FIX_PostShift_UNLESS

  return post_shift;
}

} // namespace fme_apply
