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

#include "luci/Service/CircleShapeInference.h"
#include "luci/Service/ShapeDescription.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>

#include <cassert>

namespace luci
{

ShapeDescription ShapeInference::get(loco::Node *node)
{
  // TODO Adjust indentation level
  {
    assert(loco::shape_known(node));
    return to_shape_description(loco::shape_get(node));
  }
}

} // namespace luci
