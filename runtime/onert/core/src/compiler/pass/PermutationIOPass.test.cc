/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PermutationIOPass.h"

#include "ir/Graph.h"

#include <gtest/gtest.h>

using namespace onert::ir;
using namespace onert::compiler;
using namespace onert::compiler::pass;

TEST(PermutationIOPass, type)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo graph_type{DataType::QUANT_INT8_SYMM, 1.0f, 0};
  TypeInfo actual_type{DataType::FLOAT32, 0, 0};
  auto in = graph.addOperand(shape, graph_type);
  auto out = graph.addOperand(shape, graph_type);

  // Set model input/output
  graph.addInput(in);
  graph.addOutput(out);

  // Set input/output type to float32
  CompilerOptions options;
  options.input_type.insert_or_assign(IOIndex{0}, actual_type);
  options.output_type.insert_or_assign(IOIndex{0}, actual_type);
  PermutationIOPass{graph, options}.run();

  // Check input/output type is changed to float32
  ASSERT_TRUE(graph.getInputs().at(0) != in);
  ASSERT_TRUE(graph.operands().at(graph.getInputs().at(0)).typeInfo() == actual_type);
  ASSERT_TRUE(graph.getOutputs().at(0) != out);
  ASSERT_TRUE(graph.operands().at(graph.getOutputs().at(0)).typeInfo() == actual_type);

  // Check permutation operation is added between original tensor and new tensor
  graph.operations().iterate([&](const OperationIndex &, const IOperation &op) {
    if (op.getOutputs().at(0) == in)
    {
      ASSERT_TRUE(op.getInputs().at(0) == graph.getInputs().at(0));
      ASSERT_TRUE(op.opcode() == OpCode::Permute);
    }

    if (op.getInputs().at(0) == out)
    {
      ASSERT_TRUE(op.getOutputs().at(0) == graph.getOutputs().at(0));
      ASSERT_TRUE(op.opcode() == OpCode::Permute);
    }
  });
}

TEST(PermutationIOPass, neg_type_skip)
{
  Graph graph;

  // Add tensors
  Shape shape{1, 2, 2, 1};
  TypeInfo graph_type{DataType::QUANT_INT8_SYMM, 1.0f, 0};
  TypeInfo actual_type{DataType::QUANT_INT8_SYMM, 1.0f, 0};
  auto in = graph.addOperand(shape, graph_type);
  auto out = graph.addOperand(shape, graph_type);

  // Set model input/output
  graph.addInput(in);
  graph.addOutput(out);

  // Set input/output type but same
  CompilerOptions options;
  options.input_type.insert_or_assign(IOIndex{0}, actual_type);
  options.output_type.insert_or_assign(IOIndex{0}, actual_type);
  PermutationIOPass{graph, options}.run();

  // Check input/output is same
  ASSERT_TRUE(graph.getInputs().at(0) == in);
  ASSERT_TRUE(graph.getOutputs().at(0) == out);
  ASSERT_TRUE(graph.operands().at(graph.getInputs().at(0)).typeInfo() == graph_type);
  ASSERT_TRUE(graph.operands().at(graph.getOutputs().at(0)).typeInfo() == graph_type);

  // Check no permutation
  graph.operations().iterate([&](const OperationIndex &, const IOperation &op) {
    ASSERT_TRUE(op.opcode() != OpCode::Permute);
  });
}

TEST(PermutationIOPass, layout)
{
  Graph graph;

  // Add tensors
  Shape graph_shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32, 0, 0};
  auto in = graph.addOperand(graph_shape, type);
  auto out = graph.addOperand(graph_shape, type);

  // Set model input/output
  graph.addInput(in);
  graph.addOutput(out);

  // Set input/output layout to NCHW
  CompilerOptions options;
  Layout actual_type = Layout::NCHW;
  options.input_layout.insert_or_assign(IOIndex{0}, actual_type);
  options.output_layout.insert_or_assign(IOIndex{0}, actual_type);
  PermutationIOPass{graph, options}.run();

  // Check input/output shape is changed to NCHW
  Shape actual_shape{1, 1, 2, 2};
  ASSERT_TRUE(graph.getInputs().at(0) != in);
  ASSERT_TRUE(graph.operands().at(graph.getInputs().at(0)).shape() == actual_shape);
  ASSERT_TRUE(graph.getOutputs().at(0) != out);
  ASSERT_TRUE(graph.operands().at(graph.getOutputs().at(0)).shape() == actual_shape);

  // Check permutation operation is added between original tensor and new tensor
  graph.operations().iterate([&](const OperationIndex &, const IOperation &op) {
    if (op.getOutputs().at(0) == in)
    {
      ASSERT_TRUE(op.getInputs().at(0) == graph.getInputs().at(0));
      ASSERT_TRUE(op.opcode() == OpCode::Permute);
    }

    if (op.getInputs().at(0) == out)
    {
      ASSERT_TRUE(op.getOutputs().at(0) == graph.getOutputs().at(0));
      ASSERT_TRUE(op.opcode() == OpCode::Permute);
    }
  });
}

TEST(PermutationIOPass, neg_layout_skip)
{
  Graph graph;

  // Add tensors
  Shape graph_shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32, 0, 0};
  auto in = graph.addOperand(graph_shape, type);
  auto out = graph.addOperand(graph_shape, type);

  // Set model input/output
  graph.addInput(in);
  graph.addOutput(out);

  // Set input/output layout to NHWC (same)
  CompilerOptions options;
  Layout actual_type = Layout::NHWC;
  options.input_layout.insert_or_assign(IOIndex{0}, actual_type);
  options.output_layout.insert_or_assign(IOIndex{0}, actual_type);
  PermutationIOPass{graph, options}.run();

  // Check input/output shape is changed to NCHW
  Shape actual_shape{1, 2, 2, 1};
  ASSERT_TRUE(graph.getInputs().at(0) == in);
  ASSERT_TRUE(graph.operands().at(graph.getInputs().at(0)).shape() == actual_shape);
  ASSERT_TRUE(graph.getOutputs().at(0) == out);
  ASSERT_TRUE(graph.operands().at(graph.getOutputs().at(0)).shape() == actual_shape);

  // Check no permutation
  graph.operations().iterate([&](const OperationIndex &, const IOperation &op) {
    ASSERT_TRUE(op.opcode() != OpCode::Permute);
  });
}
