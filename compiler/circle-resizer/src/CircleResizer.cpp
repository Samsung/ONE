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

#include "CircleResizer.h"

#include <mio/circle/schema_generated.h>

#include <logo/Phase.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/Importer.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>
#include <logo/RemoveDeadNodeWithQueryPass.h>
#include <luci/Import/GraphBuilderRegistry.h>

#include <luci/Log.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace circle_resizer;

namespace
{
std::vector<uint8_t> read_model(const std::string &model_path)
{
  std::ifstream file_stream(model_path, std::ios::in | std::ios::binary | std::ifstream::ate);
  if (!file_stream.is_open())
  {
    throw std::runtime_error("Failed to open file: " + model_path);
  }

  std::streamsize size = file_stream.tellg();
  file_stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file_stream.read(reinterpret_cast<char *>(buffer.data()), size))
  {
    throw std::runtime_error("Failed to read file: " + model_path);
  }

  return buffer;
}

void replace_tensor_shape(::flatbuffers::Vector<int32_t> *tensor_shape, const Shape &new_shape)
{
  const auto shape_size = tensor_shape->size();
  if (shape_size != new_shape.size())
  {
    throw std::runtime_error("Provided shape size: " + std::to_string(new_shape.size()) +
                             " is different from expected: " + std::to_string(shape_size));
  }
  for (uint32_t dim_idx = 0; dim_idx < shape_size; ++dim_idx)
  {
    tensor_shape->Mutate(dim_idx, new_shape[dim_idx].value());
  }
}

template <typename NodeType>
std::vector<Shape> extract_shapes(const std::vector<loco::Node *> &nodes)
{
  std::vector<Shape> shapes;
  for (const auto &loco_node : nodes)
  {
    shapes.push_back(Shape{});
    const auto circle_node = loco::must_cast<const NodeType *>(loco_node);
    for (uint32_t dim_idx = 0; dim_idx < circle_node->rank(); dim_idx++)
    {
      const int32_t dim_val = circle_node->dim(dim_idx).value();
      shapes.back().push_back(Dim{dim_val});
    }
  }
  return shapes;
}

} // namespace

CircleResizer::CircleResizer(const std::string &model_path) : _model_path{model_path} {}

void CircleResizer::resize_model(const std::vector<Shape> &shapes)
{
  auto model_buffer = read_model(_model_path);
  auto model = circle::GetMutableModel(model_buffer.data());
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
  if (!inputs_number == shapes.size())
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

  flatbuffers::Verifier verifier{model_buffer.data(), model_buffer.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Verification of the model failed");
  }

  const luci::GraphBuilderSource *source_ptr = &luci::GraphBuilderRegistry::get();
  luci::Importer importer(source_ptr);
  _module = importer.importModule(model_buffer.data(), model_buffer.size());

  logo::Phase phase;
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  auto graph = _module->graph();
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{graph};
  phase_runner.run(phase);
}

void CircleResizer::save_model(const std::string &output_path) const
{
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(_module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("Saving model failed");
  }
}

std::vector<Shape> CircleResizer::input_shapes() const
{
  return extract_shapes<luci::CircleInput>(loco::input_nodes(_module->graph()));
}

std::vector<Shape> CircleResizer::output_shapes() const
{
  return extract_shapes<luci::CircleOutput>(loco::output_nodes(_module->graph()));
}

CircleResizer::~CircleResizer() = default;
