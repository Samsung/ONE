/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include <memory>

#include "SharedMemoryOperands.h"

#include "ir/Graph.h"
#include "ir/operation/Permute.h"
#include "ir/operation/Squeeze.h"
#include "ir/operation/Reshape.h"

using namespace onert::backend::cpu;
using namespace onert::ir;

// Add node other than Reshape/ExpandDims/Squeeze.
// It is used for cases where Reshape input/output is not input/output on the whole model.
namespace
{
void addNotOptimizedNode(Graph *graph, const OperandIndex &input, const OperandIndex &output)
{
  graph->addOperation(std::make_unique<operation::Permute>(input, output));
}
} // namespace

TEST(SharedMemoryOperands, no_shared_memory_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4}, data_type);
  const auto not_optim_out = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, not_optim_out);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 0);
}

TEST(SharedMemoryOperands, single_reshape_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4}, data_type);
  const auto reshape_input = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, reshape_input);
  const auto reshape_output = graph->addOperand({2, 2}, data_type);
  operation::Reshape::Param shape;
  shape.new_shape = {2, 2};
  TypeInfo shape_type{DataType::INT32};
  const auto reshape_shape = graph->addOperand({2}, shape_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape_input, reshape_shape},
                                         OperandIndexSequence{reshape_output}, shape));
  const auto not_optim_out_2 = graph->addOperand({2, 2}, data_type);
  addNotOptimizedNode(graph.get(), reshape_output, not_optim_out_2);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out_2);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 1);
  EXPECT_EQ(indexes_map.begin()->first, 2);
  EXPECT_EQ(indexes_map.begin()->second, 1);
}

TEST(SharedMemoryOperands, double_reshape_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4}, data_type);
  const auto reshape1_input = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, reshape1_input);
  const auto reshape1_output = graph->addOperand({2, 2}, data_type);
  operation::Reshape::Param shape;
  shape.new_shape = {2, 2};
  TypeInfo shape_type{DataType::INT32};
  const auto reshape_shape = graph->addOperand({2}, shape_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape1_input, reshape_shape},
                                         OperandIndexSequence{reshape1_output}, shape));
  const auto reshape2_output = graph->addOperand({2, 2}, data_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape1_output, reshape_shape},
                                         OperandIndexSequence{reshape2_output}, shape));
  const auto not_optim_out_2 = graph->addOperand({2, 2}, data_type);
  addNotOptimizedNode(graph.get(), reshape2_output, not_optim_out_2);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out_2);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 2);
  auto map_it = indexes_map.begin();
  EXPECT_EQ(map_it->first, 2);
  EXPECT_EQ(map_it->second, 1);
  ++map_it;
  EXPECT_EQ(map_it->first, 4);
  EXPECT_EQ(map_it->second, 1);
}

TEST(SharedMemoryOperands, dyn_output_reshape_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4}, data_type);
  const auto reshape_input = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, reshape_input);
  const auto reshape_output = graph->addOperand({}, data_type);
  graph->operands().at(reshape_output).info().setDynamic();
  operation::Reshape::Param shape;
  TypeInfo shape_type{DataType::INT32};
  const auto reshape_shape = graph->addOperand({2}, shape_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape_input, reshape_shape},
                                         OperandIndexSequence{reshape_output}, shape));
  const auto not_optim_out_2 = graph->addOperand({}, data_type);
  addNotOptimizedNode(graph.get(), reshape_output, not_optim_out_2);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out_2);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 0);
}

TEST(SharedMemoryOperands, model_input_reshape_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto reshape_input = graph->addOperand({4}, data_type);
  const auto reshape_output = graph->addOperand({2, 2}, data_type);
  operation::Reshape::Param shape;
  shape.new_shape = {2, 2};
  TypeInfo shape_type{DataType::INT32};
  const auto reshape_shape = graph->addOperand({2}, shape_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape_input, reshape_shape},
                                         OperandIndexSequence{reshape_output}, shape));
  const auto not_optim_out = graph->addOperand({2, 2}, data_type);
  addNotOptimizedNode(graph.get(), reshape_output, not_optim_out);
  graph->addInput(reshape_input);
  graph->addOutput(not_optim_out);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 0);
}

TEST(SharedMemoryOperands, single_squeeze_graph)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4, 1}, data_type);
  const auto squeeze_input = graph->addOperand({4, 1}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, squeeze_input);
  const auto squeeze_output = graph->addOperand({4}, data_type);
  operation::Squeeze::Param axes;
  axes.dims[0] = 1;
  axes.ndim = 1;
  graph->addOperation(std::make_unique<operation::Squeeze>(
    OperandIndexSequence{squeeze_input}, OperandIndexSequence{squeeze_output}, axes));
  const auto not_optim_out_2 = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), squeeze_output, not_optim_out_2);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out_2);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 1);
  EXPECT_EQ(indexes_map.begin()->first, 2);
  EXPECT_EQ(indexes_map.begin()->second, 1);
}

TEST(SharedMemoryOperands, multiple_reshape_input_consumers)
{
  auto graph = std::make_unique<Graph>();
  TypeInfo data_type{DataType::FLOAT32};
  const auto not_optim_in = graph->addOperand({4}, data_type);
  const auto reshape_input = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), not_optim_in, reshape_input);
  const auto reshape_output = graph->addOperand({2, 2}, data_type);
  operation::Reshape::Param shape;
  shape.new_shape = {2, 2};
  TypeInfo shape_type{DataType::INT32};
  const auto reshape_shape = graph->addOperand({2}, shape_type);
  graph->addOperation(
    std::make_unique<operation::Reshape>(OperandIndexSequence{reshape_input, reshape_shape},
                                         OperandIndexSequence{reshape_output}, shape));
  const auto not_optim_out_2 = graph->addOperand({2, 2}, data_type);
  addNotOptimizedNode(graph.get(), reshape_output, not_optim_out_2);
  const auto not_optim_out_3 = graph->addOperand({4}, data_type);
  addNotOptimizedNode(graph.get(), reshape_input, not_optim_out_3);
  graph->addInput(not_optim_in);
  graph->addOutput(not_optim_out_2);
  graph->addOutput(not_optim_out_3);
  graph->verify();

  const auto indexes_map = findSharedMemoryOperandIndexes(*graph);

  ASSERT_EQ(indexes_map.size(), 1);
  EXPECT_EQ(indexes_map.begin()->first, 2);
  EXPECT_EQ(indexes_map.begin()->second, 1);
}
