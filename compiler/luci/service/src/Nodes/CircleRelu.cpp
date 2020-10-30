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

#include <luci/Service/CircleShapeSignatureInferenceRule.h>
//#include <luci/Service/CircleShapeInferenceRule.h>
//#include <luci/Service/CircleTypeInferenceRule.h>

namespace luci
{

ShapeSignature ShapeSignatureInferenceAlgorithm::visit(const luci::CircleRelu *node)
{
  auto relu_input = loco::must_cast<luci::CircleNode *>(node->features());
  return relu_input->shape_signature();
}

/*
 * How about moving other luci/Service as following example?
 *

loco::NodeShape ShapeInferenceAlgorithm::visit(const luci::CircleRelu *node)
{
  auto x_shape = loco::shape_get(node->features()).template as<loco::TensorShape>();
  return loco::NodeShape{x_shape};
}

loco::DataType TypeInferenceAlgorithm::visit(const luci::CircleRelu *node)
{
  return loco::dtype_get(node->features());
}

*/

} // namespace luci
