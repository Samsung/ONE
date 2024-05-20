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

#include "ir/train/TrainableGraph.h"
#include "ir/train/operation/BinaryArithmetic.h"
#include "ir/train/operation/ElementwiseActivation.h"
#include "ir/train/operation/FullyConnected.h"
#include "ir/train/operation/Loss.h"
#include "ir/train/LossInfo.h"

#include <gtest/gtest.h>

using namespace onert::ir;

OperationIndex addAddOperation(train::TrainableGraph &tgraph, const OperandIndexSequence inputs,
                               const OperandIndexSequence outputs)
{
  // Add "ADD" operation
  operation::BinaryArithmetic::Param param;
  param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
  param.activation = Activation::NONE;
  auto add_op = operation::BinaryArithmetic(inputs, outputs, param);
  return tgraph.addOperation(std::make_unique<train::operation::BinaryArithmetic>(add_op));
}

OperationIndex addElementwiseActivationOperation(train::TrainableGraph &tgraph,
                                                 const OperandIndexSequence inputs,
                                                 const OperandIndexSequence outputs)
{
  // Add "ElementwiseActivation" operation
  operation::ElementwiseActivation::Param param;
  auto ea_op = operation::ElementwiseActivation(inputs, outputs, param);
  return tgraph.addOperation(std::make_unique<train::operation::ElementwiseActivation>(ea_op));
}

OperationIndex addFullyConnectedOperation(train::TrainableGraph &tgraph,
                                          const OperandIndexSequence inputs,
                                          const OperandIndexSequence outputs)
{
  // Add "FullyConnected" operation
  operation::FullyConnected::Param param;
  param.weights_format = FullyConnectedWeightsFormat::Default;
  param.activation = Activation::NONE;
  auto fc_op = operation::FullyConnected(inputs, outputs, param);
  return tgraph.addOperation(std::make_unique<train::operation::FullyConnected>(fc_op));
}

OperationIndex addLossOperation(train::TrainableGraph &tgraph, const OperandIndexSequence inputs,
                                const OperandIndexSequence outputs)
{
  // Add "Loss" operation
  auto loss_op = operation::Loss(inputs, outputs);
  return tgraph.addOperation(std::make_unique<train::operation::Loss>(loss_op, train::LossInfo{}));
}

TEST(TrainableGraph, topological_sort_linear)
{
  train::TrainableGraph tgraph;

  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};

  /*
  (input) ⎼[EA]⎼> (y_pred)
                          ╲
                           [Loss]⎼> (output)
                          ╱
                  (y_true)
  */

  auto input = tgraph.addOperand(shape, type);
  auto y_pred = tgraph.addOperand(shape, type);
  auto y_true = tgraph.addOperand(shape, type);
  auto output = tgraph.addOperand(shape, type);

  tgraph.addInput({input});
  tgraph.addInput({y_true});
  tgraph.addOutput({output});

  addElementwiseActivationOperation(tgraph, {input}, {y_pred});
  addLossOperation(tgraph, {y_pred, y_true}, {output});

  EXPECT_NO_THROW(tgraph.topolSortOperations());
  EXPECT_NO_THROW(tgraph.btopolSortOperations());
}

TEST(TrainableGraph, topological_sort_nonlinear)
{
  train::TrainableGraph tgraph;

  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};

  /*
                          [EA]⎼> (lhs)
                         ╱            ╲
  (input) ⎼[EA]⎼> (split)              [Add]⎼> (y_pred)
                         ╲            ╱                ╲
                          [EA]⎼> (rhs)                  [Loss]⎼> (output)
                                                       ╱
                                               (y_true)
  */

  auto input = tgraph.addOperand(shape, type);
  auto split = tgraph.addOperand(shape, type);
  auto lhs = tgraph.addOperand(shape, type);
  auto rhs = tgraph.addOperand(shape, type);
  auto y_pred = tgraph.addOperand(shape, type);
  auto y_true = tgraph.addOperand(shape, type);
  auto output = tgraph.addOperand(shape, type);

  tgraph.addInput({input});
  tgraph.addInput({y_true});
  tgraph.addOutput({output});

  addElementwiseActivationOperation(tgraph, {input}, {split});
  addElementwiseActivationOperation(tgraph, {split}, {lhs});
  addElementwiseActivationOperation(tgraph, {split}, {rhs});
  addAddOperation(tgraph, {lhs, rhs}, {y_pred});
  addLossOperation(tgraph, {y_pred, y_true}, {output});

  EXPECT_NO_THROW(tgraph.topolSortOperations());
  EXPECT_NO_THROW(tgraph.btopolSortOperations());
}

TEST(TrainableGraph, neg_topological_sort_cycle)
{
  train::TrainableGraph tgraph;

  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};

  /*
  (input) ⎼[Add]⎼> (v) ⎼[EA]
            |            |
                         v
           (u) <⎼[EA]⎼ (y_pred)
                               ╲
                                [Loss]⎼> (output)
                               ╱
                       (y_true)
  */

  auto input = tgraph.addOperand(shape, type);
  auto u = tgraph.addOperand(shape, type);
  auto v = tgraph.addOperand(shape, type);
  auto y_pred = tgraph.addOperand(shape, type);
  auto y_true = tgraph.addOperand(shape, type);
  auto output = tgraph.addOperand(shape, type);

  tgraph.addInput({input});
  tgraph.addInput({y_true});
  tgraph.addOutput({output});

  addAddOperation(tgraph, {input, u}, {v});
  addElementwiseActivationOperation(tgraph, {v}, {y_pred});
  addElementwiseActivationOperation(tgraph, {y_pred}, {u});
  addLossOperation(tgraph, {y_pred, y_true}, {output});

  EXPECT_ANY_THROW(tgraph.topolSortOperations());
  EXPECT_ANY_THROW(tgraph.btopolSortOperations());
}

TEST(TrainableGraph, truncating_backward_topological_order_linear)
{
  train::TrainableGraph tgraph;

  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};

  /*
  (input) ⎼[EA]⎼> (u)
                     ╲
            (weight) ⎼[FC]⎼> (y_pred)
                     ╱               ╲
               (bias)                 [Loss]⎼> (output)
                                     ╱
                             (y_true)
  */

  auto input = tgraph.addOperand(shape, type);
  auto u = tgraph.addOperand(shape, type);
  auto weight = tgraph.addOperand(shape, type);
  auto bias = tgraph.addOperand(shape, type);
  auto y_pred = tgraph.addOperand(shape, type);
  auto y_true = tgraph.addOperand(shape, type);
  auto output = tgraph.addOperand(shape, type);

  tgraph.addInput({input});
  tgraph.addInput({weight});
  tgraph.addInput({bias});
  tgraph.addInput({y_true});
  tgraph.addOutput({output});

  auto ea = addElementwiseActivationOperation(tgraph, {input}, {u});
  auto fc = addFullyConnectedOperation(tgraph, {u, weight, bias}, {y_pred});
  auto loss = addLossOperation(tgraph, {y_pred, y_true}, {output});

  std::vector<OperationIndex> expected_truncation{loss, fc};
  std::vector<OperationIndex> truncation =
    tgraph.truncateBackwardOrder(tgraph.btopolSortOperations());

  ASSERT_EQ(truncation, expected_truncation);
}

TEST(TrainableGraph, truncating_backward_topological_order_nonlinear)
{
  train::TrainableGraph tgraph;

  Shape shape{1, 2, 2, 1};
  TypeInfo type{DataType::FLOAT32};

  /*
            [EA1]⎼> (u)
           ╱           ╲
          ╱  (weight1) ⎼[FC1]⎼> (v)
         ╱             ╱           ╲
        ╱       (bias1)             [Add]⎼> (y_pred)
 (input)                           ╱                ╲
        ╲                         ╱                  [Loss]⎼> (output)
         [EA2]⎼> (w)             ╱                  ╱
                    ╲           ╱           (y_true)
          (weight2) ⎼[FC2]⎼> (x)
                    ╱
            (bias2)
  */

  auto input = tgraph.addOperand(shape, type);
  auto u = tgraph.addOperand(shape, type);
  auto weight1 = tgraph.addOperand(shape, type);
  auto bias1 = tgraph.addOperand(shape, type);
  auto v = tgraph.addOperand(shape, type);
  auto w = tgraph.addOperand(shape, type);
  auto weight2 = tgraph.addOperand(shape, type);
  auto bias2 = tgraph.addOperand(shape, type);
  auto x = tgraph.addOperand(shape, type);
  auto y_pred = tgraph.addOperand(shape, type);
  auto y_true = tgraph.addOperand(shape, type);
  auto output = tgraph.addOperand(shape, type);

  tgraph.addInput({input});
  tgraph.addInput({weight1});
  tgraph.addInput({bias1});
  tgraph.addInput({weight2});
  tgraph.addInput({bias2});
  tgraph.addInput({y_true});
  tgraph.addOutput({output});

  auto ea1 = addElementwiseActivationOperation(tgraph, {input}, {u});
  auto fc1 = addFullyConnectedOperation(tgraph, {u, weight1, bias1}, {v});
  auto ea2 = addElementwiseActivationOperation(tgraph, {input}, {w});
  auto fc2 = addFullyConnectedOperation(tgraph, {w, weight2, bias2}, {x});
  auto add = addAddOperation(tgraph, {v, x}, {y_pred});
  auto loss = addLossOperation(tgraph, {y_pred, y_true}, {output});

  std::vector<OperationIndex> expected_truncation_1{loss, add, fc1, fc2};
  std::vector<OperationIndex> expected_truncation_2{loss, add, fc2, fc1};
  std::vector<OperationIndex> truncation =
    tgraph.truncateBackwardOrder(tgraph.btopolSortOperations());

  ASSERT_TRUE(truncation == expected_truncation_1 || truncation == expected_truncation_2);
}
