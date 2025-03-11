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

#include <logo/Phase.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>
#include <logo/RemoveDeadNodeWithQueryPass.h>
#include <luci/Import/GraphBuilderRegistry.h>

#include <iostream>
#include <string>

using namespace circle_resizer;

namespace
{
void replace_tensor_shape(::flatbuffers::Vector<int32_t> *tensor_shape, const Shape &new_shape)
{
  const auto shape_size = tensor_shape->size();
  if (shape_size != new_shape.size())
  {
    throw std::runtime_error("Provided shape rank: " + std::to_string(new_shape.size()) +
                             " is different from expected: " + std::to_string(shape_size));
  }
  for (uint32_t dim_idx = 0; dim_idx < shape_size; ++dim_idx)
  {
    tensor_shape->Mutate(dim_idx, new_shape[dim_idx].value());
  }
}

} // namespace

ModelEditor::ModelEditor(std::shared_ptr<ModelData> model_data) : _model_data{model_data} {}

ModelEditor &ModelEditor::resize_inputs(const std::vector<Shape> &shapes)
{
  auto model = circle::GetMutableModel(_model_data->buffer().data());
  if (!model)
  {
    throw std::runtime_error("Incorrect model format");
  }
  auto subgraphs = model->mutable_subgraphs();
  if (!subgraphs || subgraphs->size() != 1)
  {
    throw std::runtime_error("Many subgraphs are not supported");
  }
  auto subgraph = subgraphs->GetMutableObject(0);
  const auto inputs_number = subgraph->inputs()->size();
  if (inputs_number != shapes.size())
  {
    throw std::runtime_error("Expected input shapes: " + std::to_string(inputs_number) +
                             " while provided: " + std::to_string(shapes.size()));
  }
  for (int in_idx = 0; in_idx < inputs_number; ++in_idx)
  {
    const auto in_tensor_idx = subgraph->inputs()->Get(in_idx);
    auto input_shape =
      subgraph->mutable_tensors()->GetMutableObject(in_tensor_idx)->mutable_shape();
    replace_tensor_shape(input_shape, shapes[in_idx]);
  }

  // invalidate after changing input shape
  _model_data->invalidate_module();

  logo::Phase phase;
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  auto graph = _model_data->module()->graph();
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{graph};
  phase_runner.run(phase);

  // invalidate after shape inference
  _model_data->invalidate_buffer();

  return *this;
}
