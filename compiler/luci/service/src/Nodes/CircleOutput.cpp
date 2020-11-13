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

#include <luci/Service/CircleShapeSignatureInference.h>
//#include <luci/Service/CircleShapeInferenceRule.h>
//#include <luci/Service/CircleTypeInferenceRule.h>

namespace luci
{

ShapeSignature ssinf::Algorithm::visit(const luci::CircleOutput *node)
{
  return ssinf::signature_of_input(node, 0);
}

/*
 * How about moving other luci/Service as following example?
 *

loco::NodeShape ShapeInferenceAlgorithm::visit(const luci::CircleOutput *node)
{
  auto graph_outputs = node->graph()->outputs();
  auto graph_output = graph_outputs->at(node->index());
  auto output_shape = graph_output->shape();

  return loco::NodeShape{*output_shape};
}

loco::DataType TypeInferenceAlgorithm::visit(const luci::CircleOutput *node)
{
    auto graph_outputs = node->graph()->outputs();
    auto graph_output = graph_outputs->at(node->index());
    auto output_dtype = graph_output->dtype();

    if (dynamic_cast<luci::CircleOutputDummy *>(node->from()) == nullptr &&
        dynamic_cast<luci::CircleOutputExclude *>(node->from()) == nullptr)
    {
      // We don't care for the type if from() is CircleOutputDummy or CircleOutputExclude
      // from() type should match that of CircleOutput
      assert(output_dtype == loco::dtype_get(node->from()));
    }
    return output_dtype;
}

*/

} // namespace luci
