/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Service/CircleTypeInference.h"
#include "CircleTypeInferenceHelper.h"

#include <luci/Log.h>

#include <loco.h>

#include <type_traits>

namespace
{

bool inputs_dtype_ready(const luci::CircleNode *node)
{
  for (uint32_t arity = 0; arity < node->arity(); ++arity)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->arg(arity));
    if (input_node->dtype() == loco::DataType::Unknown)
      return false;
  }

  return true;
}

} // namespace

namespace luci
{
namespace tinf
{

bool Rule::infer(const luci::CircleNode *circle_node, loco::DataType &dtype) const
{
  LOGGER(l);
  VERBOSE(l, 1) << "[CircleTypeInference] " << circle_node->name();
  VERBOSE(l, 1) << "  before: " << static_cast<int>(circle_node->dtype());

  if (!inputs_dtype_ready(circle_node))
  {
    VERBOSE(l, 1) << "   after: Some inputs are not ready for inference";
    return false;
  }

  Algorithm alg;
  dtype = circle_node->accept(&alg);

  VERBOSE(l, 1) << "   after: " << static_cast<int>(dtype);

  return true;
}

} // namespace tinf
} // namespace luci
