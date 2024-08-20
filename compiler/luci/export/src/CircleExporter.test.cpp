/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/CircleExporter.h"

#include <luci/Plan/CircleNodeExecutionPlan.h>
#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/IR/Nodes/CircleRelu.h>
#include <luci/UserSettings.h>

#include <mio/circle/schema_generated.h>
#include <flatbuffers/flatbuffers.h>

#include <gtest/gtest.h>

class SampleGraphContract : public luci::CircleExporter::Contract
{
public:
  SampleGraphContract() : luci::CircleExporter::Contract(), _buffer(new std::vector<char>)
  {
    // create needed entities
    auto g = loco::make_graph();
    auto graph_input = g->inputs()->create();
    auto graph_output = g->outputs()->create();
    input_node = g->nodes()->create<luci::CircleInput>();
    output_node = g->nodes()->create<luci::CircleOutput>();
    relu_node = g->nodes()->create<luci::CircleRelu>();

    // link nodes and link them to graph
    relu_node->features(input_node);
    output_node->from(relu_node);
    input_node->index(graph_input->index());
    output_node->index(graph_output->index());

    // Set needed properties
    input_node->name("input");
    output_node->name("output");
    relu_node->name("relu");
    input_node->dtype(loco::DataType::FLOAT32);

    graph_input->shape({1, 2, 3, 4});
    graph_input->dtype(loco::DataType::FLOAT32);

    graph_output->shape({1, 2, 3, 4});
    graph_output->dtype(loco::DataType::FLOAT32);

    _m = std::unique_ptr<luci::Module>{new luci::Module};
    _m->add(std::move(g));
  }

  luci::Module *module(void) const override { return _m.get(); }

public:
  bool store(const char *ptr, const size_t size) const override
  {
    _buffer->resize(size);
    std::copy(ptr, ptr + size, _buffer->begin());
    return true;
  }

  const std::vector<char> &get_buffer() { return *_buffer; }

public:
  luci::CircleInput *input_node;
  luci::CircleOutput *output_node;
  luci::CircleRelu *relu_node;

private:
  std::unique_ptr<luci::Module> _m;
  std::unique_ptr<std::vector<char>> _buffer;
};

TEST(CircleExport, export_execution_plan)
{
  SampleGraphContract contract;
  uint32_t reference_order = 1;
  uint32_t reference_offset = 100u;
  luci::add_execution_plan(contract.relu_node,
                           luci::CircleNodeExecutionPlan(reference_order, {reference_offset}));

  luci::UserSettings::settings()->set(luci::UserSettings::ExecutionPlanGen, true);
  luci::CircleExporter exporter;

  exporter.invoke(&contract);

  ASSERT_FALSE(contract.get_buffer().empty());
  std::unique_ptr<circle::ModelT> model(circle::GetModel(contract.get_buffer().data())->UnPack());
  ASSERT_NE(model.get(), nullptr);
  ASSERT_EQ(model->metadata[0]->name, "ONE_execution_plan_table");
  auto metadata_buffer = model->metadata[0]->buffer;
  auto &buffer = model->buffers[metadata_buffer]->data;
  ASSERT_EQ(buffer.size(), 20);
  uint32_t *raw_table_contents = reinterpret_cast<uint32_t *>(buffer.data());

  auto num_entries = raw_table_contents[0];
  ASSERT_EQ(num_entries, 1);
  auto node_id = raw_table_contents[1];
  ASSERT_EQ(node_id, 1); // relu node is second (aka id 1) in tological sort in exporter
  auto node_plan_size = raw_table_contents[2];
  ASSERT_EQ(node_plan_size, 2); // 1 for execution order, 1 for memory offset value
  auto node_plan_order = raw_table_contents[3];
  ASSERT_EQ(node_plan_order,
            reference_order); // this value goes from CircleNodeExecutionPlan initialization
  auto node_plan_offset = raw_table_contents[4];
  ASSERT_EQ(node_plan_offset,
            reference_offset); // this value goes from CircleNodeExecutionPlan initialization
}

TEST(CircleExport, export_execution_plan_nosetting_NEG)
{
  SampleGraphContract contract;
  uint32_t reference_order = 1;
  uint32_t reference_offset = 100u;
  luci::add_execution_plan(contract.relu_node,
                           luci::CircleNodeExecutionPlan(reference_order, {reference_offset}));

  luci::UserSettings::settings()->set(luci::UserSettings::ExecutionPlanGen, false);
  luci::CircleExporter exporter;

  exporter.invoke(&contract);

  ASSERT_FALSE(contract.get_buffer().empty());
  std::unique_ptr<circle::ModelT> model(circle::GetModel(contract.get_buffer().data())->UnPack());
  ASSERT_NE(model.get(), nullptr);
  ASSERT_EQ(model->metadata.size(), 0);
}
