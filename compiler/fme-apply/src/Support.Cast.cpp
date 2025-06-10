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

luci::CircleCustom *to_scale(loco::Node *node)
{
  auto scale = dynamic_cast<luci::CircleCustom *>(node);
  if (not scale)
    return nullptr;

  if (scale->custom_code() != "scale")
    return nullptr;

  // TODO Return false?
  assert(scale->numInputs() == 2); // FIX_PreScale_UNLESS

  return scale;
}

} // namespace fme_apply
