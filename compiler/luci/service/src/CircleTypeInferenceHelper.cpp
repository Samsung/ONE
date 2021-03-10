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

#include "CircleTypeInferenceHelper.h"

#include <loco/Service/TypeInference.h>

namespace luci
{

loco::DataType dtype_get(const loco::Node *node)
{
  assert(luci::dtype_known(node));
  return loco::must_cast<const luci::CircleNode *>(node)->dtype();
}

bool dtype_known(const loco::Node *node)
{
  return loco::must_cast<const luci::CircleNode *>(node)->dtype() != loco::DataType::Unknown;
}

} // namespace luci

namespace luci
{
namespace tinf
{

// Helper function will be added

} // namespace tinf
} // namespace luci
