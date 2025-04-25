/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelEditor.h"

#include <mio/circle/schema_generated.h>

#include <loco/IR/Graph.h>
#include <logo/Phase.h>
#include <logo/RemoveDeadNodeWithQueryPass.h>
#include <luci/IR/Nodes/CircleInput.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>

using namespace circle_resizer;

namespace
{

void change_single_input_shape(luci::CircleInput *circle_input, const Shape &new_shape)
{
  circle_input->rank(new_shape.rank());
  for (uint32_t i = 0; i < new_shape.rank(); ++i)
  {
    if (new_shape[i].is_dynamic())
    {
      circle_input->dim(i) = loco::Dimension(); // empty ctor means dynamic dimension
    }
    else
    {
      // a value here can be in range (0, std::numeric_limits<int32_t>::max()) so the cast is safe
      circle_input->dim(i) = loco::Dimension(static_cast<uint32_t>(new_shape[i].value()));
    }
  }
}

void change_inputs_shapes(loco::Graph *graph, const std::vector<Shape> &new_inputs_shapes)
{
  auto graph_inputs = loco::input_nodes(graph);
  if (graph_inputs.size() != new_inputs_shapes.size())
  {
    throw std::runtime_error("Expected " + std::to_string(graph_inputs.size()) +
                             " shapes but provided " + std::to_string(new_inputs_shapes.size()));
  }
  for (size_t in_idx = 0; in_idx < new_inputs_shapes.size(); ++in_idx)
  {
    auto circle_input = loco::must_cast<luci::CircleInput *>(graph_inputs[in_idx]);
    change_single_input_shape(circle_input, new_inputs_shapes[in_idx]);
  }
}

} // namespace

ModelEditor::ModelEditor(std::shared_ptr<CircleModel> circle_model) : _circle_model{circle_model}
{
  assert(circle_model != nullptr); // FIX_CALLER_UNLESS
}

ModelEditor &ModelEditor::resize_inputs(const std::vector<Shape> &new_inputs_shapes)
{
  auto graph = _circle_model->module()->graph();
  change_inputs_shapes(graph, new_inputs_shapes);

  logo::Phase phase;
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{graph};
  try
  {
    phase_runner.run(phase);
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error("Exception during resizing with message: " + std::string{e.what()});
  }

  return *this;
}
