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

#include <luci/Service/CircleShapeInference.h>
#include <luci/Service/CircleShapeSignatureInference.h>
#include <luci/Service/CircleTypeInference.h>

namespace luci
{

loco::TensorShape sinf::Algorithm::visit(const luci::CircleOutput *node)
{
  auto graph_output = node->graph()->outputs()->at(node->index());
  return *(graph_output->shape());
}

ShapeSignature ssinf::Algorithm::visit(const luci::CircleOutput *node)
{
  return input_arg_signature(node, 0);
}

loco::DataType tinf::Algorithm::visit(const luci::CircleOutput *node)
{
  auto graph_output = node->graph()->outputs()->at(node->index());
  auto output_dtype = graph_output->dtype();

  return output_dtype;
}

} // namespace luci
