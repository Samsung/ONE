/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <luci/Service/CircleTypeInference.h>

namespace
{

struct CircleIfOutGraphs
{
  loco::GraphOutput *then_graph_output;
  loco::GraphOutput *else_graph_output;
};

} // namespace

namespace
{

CircleIfOutGraphs get_out_graphs(const luci::CircleIfOut *node)
{
  CircleIfOutGraphs ret_out;

  /**
   * @note  IF operator type and shape are that of the "then" and "else"
   *        Graph Outputs.
   */
  auto circle_if = loco::must_cast<const luci::CircleIf *>(node->input());

  auto index = node->index();
  auto then_graph = circle_if->then_graph();
  auto else_graph = circle_if->else_graph();
  assert(then_graph != nullptr);
  assert(else_graph != nullptr);

  // shape and type are assumed to be same
  // these are checked at post_import_graph() in Import
  auto then_outputs = loco::output_nodes(then_graph);
  auto else_outputs = loco::output_nodes(else_graph);
  assert(then_outputs.size() == else_outputs.size());
  assert(index < static_cast<int32_t>(then_outputs.size()));

  auto then_out = loco::must_cast<luci::CircleOutput *>(then_outputs.at(index));
  auto else_out = loco::must_cast<luci::CircleOutput *>(else_outputs.at(index));

  auto then_graph_outputs = then_graph->outputs(); // loco::GraphOutput items
  auto else_graph_outputs = else_graph->outputs();
  assert(then_graph_outputs->size() == else_graph_outputs->size());

  ret_out.then_graph_output = then_graph_outputs->at(then_out->index());
  ret_out.else_graph_output = else_graph_outputs->at(else_out->index());

  return ret_out;
}

} // namespace

namespace luci
{

loco::TensorShape sinf::Algorithm::visit(const luci::CircleIfOut *node)
{
  auto graphs = get_out_graphs(node);
  assert(*graphs.then_graph_output->shape() == *graphs.else_graph_output->shape());
  return *graphs.then_graph_output->shape();
}

loco::DataType tinf::Algorithm::visit(const luci::CircleIfOut *node)
{
  auto graphs = get_out_graphs(node);
  assert(graphs.then_graph_output->dtype() == graphs.else_graph_output->dtype());
  return graphs.then_graph_output->dtype();
}

} // namespace luci
