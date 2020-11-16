/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Graph.h"
#include "ir/Index.h"
#include "ir/OperandIndexSequence.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/Concat.h"

#include <memory>

#include <stdexcept>

using Index = onert::ir::IOIndex;
using IndexSet = onert::ir::OperandIndexSequence;

TEST(ir_Operation_setIO, operation_setIO_conv)
{
  onert::ir::Graph graph{0};

  onert::ir::Shape shape{3};
  onert::ir::TypeInfo type{onert::ir::DataType::INT32};

  // Add Conv
  using Graph = onert::ir::operation::Conv2D;

  auto input_operand = graph.addOperand(shape, type);
  auto kernel_operand = graph.addOperand(shape, type);
  auto bias_operand = graph.addOperand(shape, type);
  IndexSet inputs{input_operand, kernel_operand, bias_operand};

  Graph::Param conv_params;
  conv_params.padding.type = onert::ir::PaddingType::SAME;
  conv_params.stride.horizontal = 1;
  conv_params.stride.vertical = 1;
  conv_params.activation = onert::ir::Activation::NONE;

  auto output_operand = graph.addOperand(shape, type).value();
  IndexSet outputs{output_operand};

  auto conv = std::make_unique<Graph>(inputs, outputs, conv_params);

  ASSERT_NE(conv, nullptr);
  ASSERT_EQ(conv->getInputs().at(Index{0}).value(), inputs.at(0).value());
  conv->setInputs({8, 9, 10});
  ASSERT_NE(conv->getInputs().at(Index{0}).value(), inputs.at(0).value());
  ASSERT_EQ(conv->getInputs().at(Index{0}).value(), 8);
}

TEST(ir_Operation_setIO, neg_operation_setIO_concat)
{
  onert::ir::Graph graph{0};

  onert::ir::Shape shape{3};

  onert::ir::TypeInfo type{onert::ir::DataType::INT32};

  using Graph = onert::ir::operation::Concat;

  // Add Concat
  IndexSet inputs;
  for (int i = 0; i < 6; ++i)
  {
    inputs.append(graph.addOperand(shape, type));
  }

  Graph::Param concat_params{0};

  auto output_operand = graph.addOperand(shape, type).value();
  IndexSet outputs{output_operand};

  auto concat = std::make_unique<Graph>(inputs, outputs, concat_params);

  ASSERT_NE(concat, nullptr);
  ASSERT_EQ(concat->getInputs().size(), 6);
  ASSERT_EQ(concat->getInputs().at(Index{0}).value(), inputs.at(0).value());

  concat->setInputs({80, 6, 9, 11});
  ASSERT_EQ(concat->getInputs().size(), 4);
  ASSERT_NE(concat->getInputs().at(Index{0}).value(), inputs.at(0).value());
  ASSERT_EQ(concat->getInputs().at(Index{0}).value(), 80);
  ASSERT_EQ(concat->getInputs().at(Index{2}).value(), 9);
  ASSERT_THROW(concat->getInputs().at(Index{5}), std::out_of_range);
}
