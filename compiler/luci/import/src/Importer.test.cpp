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

#include "luci/Importer.h"

#include <luci/IR/CircleNode.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>

#include <gtest/gtest.h>
#include <mio/circle/schema_generated.h>
#include <flatbuffers/flatbuffers.h>

TEST(CircleImport, Dummy)
{
  luci::Importer import;

  SUCCEED();
}

// helpers for flatbuffers
namespace
{

struct BasicCircleModel
{
  std::unique_ptr<circle::ModelT> model;

  BasicCircleModel()
  {
    model = std::make_unique<circle::ModelT>();
    model->buffers.push_back(std::make_unique<circle::BufferT>());
    model->description = "nnpackage";
    model->version = 0;
  }

  uint32_t add_subgraph()
  {
    model->subgraphs.push_back(std::make_unique<circle::SubGraphT>());
    model->subgraphs.back()->name = "";
    return model->subgraphs.size() - 1;
  }

  void add_subgraph_inputs(uint32_t subgraph_id, const std::vector<uint32_t> &inputs)
  {
    model->subgraphs[subgraph_id]->inputs.assign(inputs.begin(), inputs.end());
  }

  void add_subgraph_outputs(uint32_t subgraph_id, const std::vector<uint32_t> &outputs)
  {
    model->subgraphs[subgraph_id]->outputs.assign(outputs.begin(), outputs.end());
  }

  uint32_t add_builtin_opcode(circle::BuiltinOperator opcode)
  {
    uint32_t id = model->operator_codes.size();
    model->operator_codes.push_back(std::make_unique<circle::OperatorCodeT>());
    model->operator_codes[id]->deprecated_builtin_code = opcode;
    model->operator_codes[id]->builtin_code = opcode;
    model->operator_codes[id]->version = 1;
    return id;
  }

  uint32_t add_buffer()
  {
    model->buffers.push_back(std::make_unique<circle::BufferT>());
    return model->buffers.size() - 1;
  }

  uint32_t add_float_tensor(uint32_t graph_id, const std::vector<int32_t> &shape,
                            uint32_t buffer_id)
  {
    auto &graph = model->subgraphs[graph_id];
    uint32_t idx = graph->tensors.size();
    graph->tensors.push_back(std::make_unique<circle::TensorT>());
    graph->tensors[idx]->shape = shape;
    graph->tensors[idx]->type = circle::TensorType_FLOAT32;
    graph->tensors[idx]->buffer = buffer_id;
    graph->tensors[idx]->name = std::to_string(idx);
    graph->tensors[idx]->quantization.reset(nullptr);
    graph->tensors[idx]->is_variable = false;
    graph->tensors[idx]->sparsity.reset(nullptr);
    (void)graph->tensors[idx]->shape_signature;
    return idx;
  }

  uint32_t add_builtin_operator(uint32_t graph_id, uint32_t opcode_id,
                                const std::vector<uint32_t> &inputs,
                                const std::vector<uint32_t> &outputs)
  {
    auto &graph = model->subgraphs[graph_id];
    auto idx = graph->operators.size();
    graph->operators.push_back(std::make_unique<circle::OperatorT>());
    graph->operators[idx]->opcode_index = opcode_id;
    graph->operators[idx]->inputs.assign(inputs.begin(), inputs.end());
    graph->operators[idx]->outputs.assign(outputs.begin(), outputs.end());
    graph->operators[idx]->builtin_options.Reset();
    (void)graph->operators[idx]->custom_options;
    graph->operators[idx]->custom_options_format = circle::CustomOptionsFormat_FLEXBUFFERS;
    (void)graph->operators[idx]->mutating_variable_inputs;
    (void)graph->operators[idx]->intermediates;
    return idx;
  }

  uint32_t add_plan_metadata(uint32_t buffer_id)
  {
    static_assert(sizeof(uint32_t) == 4, "metadata is stored in blocks of 32 bit unsiged ints");
    uint32_t idx = model->metadata.size();
    model->metadata.push_back(std::make_unique<circle::MetadataT>());
    model->metadata[idx]->name = "ONE_execution_plan_table";
    model->metadata[idx]->buffer = buffer_id;
    model->buffers[buffer_id]->data.resize(4);
    auto &entries_count = *reinterpret_cast<uint32_t *>(model->buffers[buffer_id]->data.data());
    entries_count = 0;
    return idx;
  }

  void add_plan_entry(uint32_t plan_buffer_id, uint32_t execution_order,
                      const std::vector<uint32_t> &offsets)
  {
    auto &buffer = model->buffers[plan_buffer_id]->data;
    auto old_size = buffer.size();
    assert(old_size % 4 == 0);
    assert(old_size > 0);

    // Allocate space for new entry:
    // 4 bytes for entry id
    // 4 bytes for entry size
    // 4 bytes for execution order
    // offsets.size() * 4 bytes for offsets
    buffer.resize(old_size + 12 + offsets.size() * 4);
    uint32_t *number_of_entries_ptr = reinterpret_cast<uint32_t *>(buffer.data());
    *number_of_entries_ptr += 1;

    uint32_t *entry_data_ptr = reinterpret_cast<uint32_t *>(buffer.data() + old_size);

    entry_data_ptr[0] = *number_of_entries_ptr - 1; // entry id
    entry_data_ptr[1] = 1 + offsets.size();         // entry size
    entry_data_ptr[2] = execution_order;            // execution order
    std::copy(offsets.begin(), offsets.end(), entry_data_ptr + 3);
  }
};

struct SimpleRELUModel : public BasicCircleModel
{
  SimpleRELUModel()
  {
    auto relu_opcode_id = add_builtin_opcode(circle::BuiltinOperator_RELU);

    uint32_t subgraph_id = add_subgraph();

    auto input_buffer_id = add_buffer();
    auto output_buffer_id = add_buffer();

    auto input_tensor_idx = add_float_tensor(subgraph_id, {1, 2, 3, 4}, input_buffer_id);
    auto output_tensor_idx = add_float_tensor(subgraph_id, {1, 2, 3, 4}, output_buffer_id);

    add_subgraph_inputs(subgraph_id, {input_tensor_idx});
    add_subgraph_outputs(subgraph_id, {output_tensor_idx});

    add_builtin_operator(subgraph_id, relu_opcode_id, {0}, {1});
  }
};

} // namespace

/**
 * This test checks that one op RELU model with execution plan is successfully imported
 */
TEST(CircleImport, simple_plan)
{
  SimpleRELUModel model;
  auto metadata_buffer_id = model.add_buffer();
  model.add_plan_metadata(metadata_buffer_id);

  model.add_plan_entry(metadata_buffer_id, 1, {100});
  model.add_plan_entry(metadata_buffer_id, 2, {300});
  model.add_plan_entry(metadata_buffer_id, 3, {200});

  flatbuffers::FlatBufferBuilder fbb;
  auto model_offset = circle::Model::Pack(fbb, model.model.get(), nullptr);
  circle::FinishModelBuffer(fbb, model_offset);

  auto model_ptr = circle::GetModel(fbb.GetBufferPointer());
  luci::Importer import;

  auto luci_module = import.importModule(model_ptr);

  auto main_graph = luci_module->graph();
  for (int i = 0; i < main_graph->nodes()->size(); ++i)
  {
    auto node = loco::must_cast<luci::CircleNode *>(main_graph->nodes()->at(i));
    switch (node->opcode())
    {
      case luci::CircleOpcode::CIRCLEINPUT:
      {
        ASSERT_TRUE(luci::has_execution_plan(node));
        auto plan = luci::get_execution_plan(node);
        ASSERT_EQ(plan.order_in_plan(), 1);
        ASSERT_EQ(plan.offsets().size(), 1);
        ASSERT_EQ(plan.offsets()[0], 100);
        break;
      }
      case luci::CircleOpcode::CIRCLEOUTPUT:
      {
        ASSERT_TRUE(luci::has_execution_plan(node));
        auto plan = luci::get_execution_plan(node);
        ASSERT_EQ(plan.order_in_plan(), 3);
        ASSERT_EQ(plan.offsets().size(), 1);
        ASSERT_EQ(plan.offsets()[0], 200);
        break;
      }
      case luci::CircleOpcode::RELU:
      {
        ASSERT_TRUE(luci::has_execution_plan(node));
        auto plan = luci::get_execution_plan(node);
        ASSERT_EQ(plan.order_in_plan(), 2);
        ASSERT_EQ(plan.offsets().size(), 1);
        ASSERT_EQ(plan.offsets()[0], 300);
        break;
      }
      default:
        FAIL();
    }
  }
}

/**
 * This test checks that model with incomplete execution plan is successfully imported
 */
TEST(CircleImport, incomplete_plan_NEG)
{
  SimpleRELUModel model;
  auto metadata_buffer_id = model.add_buffer();
  model.add_plan_metadata(metadata_buffer_id);

  model.add_plan_entry(metadata_buffer_id, 1, {100});

  flatbuffers::FlatBufferBuilder fbb;
  auto model_offset = circle::Model::Pack(fbb, model.model.get(), nullptr);
  circle::FinishModelBuffer(fbb, model_offset);

  auto model_ptr = circle::GetModel(fbb.GetBufferPointer());
  luci::Importer import;

  auto luci_module = import.importModule(model_ptr);

  auto main_graph = luci_module->graph();
  for (int i = 0; i < main_graph->nodes()->size(); ++i)
  {
    auto node = loco::must_cast<luci::CircleNode *>(main_graph->nodes()->at(i));
    switch (node->opcode())
    {
      case luci::CircleOpcode::CIRCLEINPUT:
      {
        ASSERT_TRUE(luci::has_execution_plan(node));
        auto plan = luci::get_execution_plan(node);
        ASSERT_EQ(plan.order_in_plan(), 1);
        ASSERT_EQ(plan.offsets().size(), 1);
        ASSERT_EQ(plan.offsets()[0], 100);
        break;
      }
      case luci::CircleOpcode::CIRCLEOUTPUT:
      case luci::CircleOpcode::RELU:
      {
        ASSERT_FALSE(luci::has_execution_plan(node));
        break;
      }
      default:
        FAIL();
    }
  }
}

/**
 * This test checks that corrupted execution plan induce exception
 */
TEST(CircleImport, corrupted_plan_NEG)
{
  SimpleRELUModel model;
  auto metadata_buffer_id = model.add_buffer();
  model.add_plan_metadata(metadata_buffer_id);

  model.add_plan_entry(metadata_buffer_id, 1, {100});
  model.add_plan_entry(metadata_buffer_id, 2, {300});
  model.add_plan_entry(metadata_buffer_id, 3, {200});

  // corrupt data
  *reinterpret_cast<uint32_t *>(model.model->buffers[metadata_buffer_id]->data.data()) = 4;

  flatbuffers::FlatBufferBuilder fbb;
  auto model_offset = circle::Model::Pack(fbb, model.model.get(), nullptr);
  circle::FinishModelBuffer(fbb, model_offset);

  auto model_ptr = circle::GetModel(fbb.GetBufferPointer());
  luci::Importer import;

  ASSERT_ANY_THROW(import.importModule(model_ptr));
}

/**
 * This test checks that empty execution plan entry induce exception
 */
TEST(CircleImport, corrupted_plan_entry_NEG)
{
  SimpleRELUModel model;
  auto metadata_buffer_id = model.add_buffer();
  model.add_plan_metadata(metadata_buffer_id);

  model.add_plan_entry(metadata_buffer_id, 1, {100});

  // add corrupted entry with 0 size
  {
    auto &buffer = model.model->buffers[metadata_buffer_id]->data;
    auto old_size = buffer.size();

    // Allocate space for new entry:
    // 4 bytes for entry id
    // 4 bytes for entry size
    buffer.resize(old_size + 8);
    uint32_t *number_of_entries_ptr = reinterpret_cast<uint32_t *>(buffer.data());
    *number_of_entries_ptr += 1;

    uint32_t *entry_data_ptr = reinterpret_cast<uint32_t *>(buffer.data() + old_size);

    entry_data_ptr[0] = *number_of_entries_ptr - 1; // entry id
    entry_data_ptr[1] = 0;                          // entry size
  }

  model.add_plan_entry(metadata_buffer_id, 3, {200});

  flatbuffers::FlatBufferBuilder fbb;
  auto model_offset = circle::Model::Pack(fbb, model.model.get(), nullptr);
  circle::FinishModelBuffer(fbb, model_offset);

  auto model_ptr = circle::GetModel(fbb.GetBufferPointer());
  luci::Importer import;

  ASSERT_ANY_THROW(import.importModule(model_ptr));
}
