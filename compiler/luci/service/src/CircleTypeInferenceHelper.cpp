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

#include "luci/Service/CircleTypeInferenceHelper.h"

namespace luci
{

namespace tinf
{

loco::DataType input_arg_dtype(const luci::CircleNode *node, unsigned int index)
{
  if (node->arity() <= index)
    throw std::runtime_error("Arity index out of range");

  auto input_node = loco::must_cast<luci::CircleNode *>(node->arg(index));
  return input_node->dtype();
}

} // namespace tinf

} // namespace luci
