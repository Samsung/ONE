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

#include "UseDefGenerator.h"

#include "ir/train/TrainableGraph.h"
#include "ir/train/Operations.Include.h"

#include <gtest/gtest.h>

namespace
{

using namespace onert::ir;

OperationIndex addConv2DOperation(train::TrainableGraph &tgraph, const OperandIndexSequence inputs,
                                  const OperandIndexSequence outputs)
{
  // Add "Conv2D" operation
  operation::Conv2D::Param param;
  param.padding = PaddingType::SAME;
  param.dilation = Dilation{1, 1};
  param.stride = Stride{1, 1};
  param.activation = Activation::NONE;
  auto conv2d_op = operation::Conv2D(inputs, outputs, param);
  return tgraph.addOperation(std::make_unique<train::operation::Conv2D>(conv2d_op));
}

OperationIndex addDepthwiseConv2DOperation(train::TrainableGraph &tgraph,
                                           const OperandIndexSequence inputs,
                                           const OperandIndexSequence outputs)
{
  // Add "DepthwiseConv2D" operation
  operation::DepthwiseConv2D::Param param;
  param.padding = PaddingType::SAME;
  param.dilation = Dilation{1, 1};
  param.stride = Stride{1, 1};
  param.activation = Activation::NONE;
  param.multiplier = 1.0f;
  auto conv2d_op = operation::DepthwiseConv2D(inputs, outputs, param);
  return tgraph.addOperation(std::make_unique<train::operation::DepthwiseConv2D>(conv2d_op));
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
  const auto &y_pred_index = inputs.at(0);
  const auto &y_pred = tgraph.operands().at(y_pred_index);
  const auto &y_pred_node = tgraph.operations().at(y_pred.getDef());
  const auto y_pred_op_code = y_pred_node.opcode();

  auto loss_op = operation::Loss(inputs, outputs);
  return tgraph.addOperation(
    std::make_unique<train::operation::Loss>(loss_op, train::LossInfo{}, y_pred_op_code));
}

train::UseDefChain createUseDefChain(const Operand &operand,
                                     std::vector<train::TrainingOperationIndex> uses,
                                     std::vector<train::TrainingOperationIndex> defs)
{
  train::UseDefChain usedefs{operand};

  for (const auto &use : uses)
    usedefs.insertTrainingUse(use);

  for (const auto &def : defs)
    usedefs.insertTrainingDef(def);

  return usedefs;
}

void enableAllBackwarding(train::TrainableGraph &tgraph)
{
  tgraph.operations().iterate(
    [&](const OperationIndex &index, const IOperation &) { tgraph.enableBackward(index); });
}

} // namespace

TEST(UseDefGenerator, one_op)
{
  // BinaryArtihmetic - Add
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (ba_input1) ⎼[BA]⎼> (y_pred)
             ╱                   ╱               ╲
    (weights)        (ba_input2)                  [Loss]⎼> (output)
                                                 ╱
                                         (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto ba_input1 = tgraph.addOperand(shape, type);
    const auto ba_input2 = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));
    tgraph.operands().at(ba_input2).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {ba_input1});

    operation::BinaryArithmetic::Param param;
    param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    param.activation = Activation::NONE;
    const auto ba_op = operation::BinaryArithmetic({ba_input1, ba_input2}, {y_pred}, param);
    const auto ba_index =
      tgraph.addOperation(std::make_unique<train::operation::BinaryArithmetic>(ba_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_ba_index{ba_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_ba_index{ba_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_ba_input1_uses{
      forwarding_ba_index, backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_ba_input1_def{
      forwarding_fc_index};
    const auto expected_forwarding_ba_input1 =
      createUseDefChain(tgraph.operands().at(ba_input1), expected_forwarding_ba_input1_uses,
                        expected_forwarding_ba_input1_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_ba_input2_uses{
      forwarding_ba_index, backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_ba_input2_def{};
    const auto expected_forwarding_ba_input2 =
      createUseDefChain(tgraph.operands().at(ba_input2), expected_forwarding_ba_input2_uses,
                        expected_forwarding_ba_input2_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{forwarding_ba_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_ba_input1_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_ba_input1_def{
      backwarding_ba_index};
    const auto expected_backwarding_ba_input1 =
      createUseDefChain(tgraph.operands().at(ba_input1), expected_backwarding_ba_input1_uses,
                        expected_backwarding_ba_input1_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_ba_input2_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_ba_input2_def{};
    const auto expected_backwarding_ba_input2 =
      createUseDefChain(tgraph.operands().at(ba_input2), expected_backwarding_ba_input2_uses,
                        expected_backwarding_ba_input2_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_ba_input1 ==
                training_usedefs.at(train::TrainingOperandIndex{ba_input1, true}));
    EXPECT_TRUE(expected_forwarding_ba_input2 ==
                training_usedefs.at(train::TrainingOperandIndex{ba_input2, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_ba_input1 ==
                training_usedefs.at(train::TrainingOperandIndex{ba_input1, false}));
    EXPECT_TRUE(expected_backwarding_ba_input2 ==
                training_usedefs.at(train::TrainingOperandIndex{ba_input2, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Conv2D
  {
    train::TrainableGraph tgraph;

    Shape shape{1, 2, 2, 1};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[Conv2D]⎼> (y_pred)
             ╱                   ╲
    (weights)                     [Loss]⎼> (output)
                                 ╱
                         (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto conv2d_index =
      addConv2DOperation(tgraph, {input, weights, OperandIndex{}}, {y_pred});
    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_conv2d_index{conv2d_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_conv2d_index{conv2d_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{
      forwarding_conv2d_index, backwarding_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_conv2d_index, backwarding_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_conv2d_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{
      backwarding_conv2d_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_conv2d_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // DepthwiseConv2D
  {
    train::TrainableGraph tgraph;

    Shape shape{1, 2, 2, 1};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[DepthwiseConv2D]⎼> (y_pred)
             ╱                            ╲
    (weights)                              [Loss]⎼> (output)
                                          ╱
                                  (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto depthwise_conv2d_index =
      addDepthwiseConv2DOperation(tgraph, {input, weights, OperandIndex{}}, {y_pred});
    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_depthwise_conv2d_index{depthwise_conv2d_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_depthwise_conv2d_index{depthwise_conv2d_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{
      forwarding_depthwise_conv2d_index, backwarding_depthwise_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_depthwise_conv2d_index, backwarding_depthwise_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_depthwise_conv2d_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{
      backwarding_depthwise_conv2d_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_depthwise_conv2d_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_depthwise_conv2d_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // ElementwiseActivation
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (ea_input) ⎼[EA]⎼> (y_pred)
             ╱                                  ╲
    (weights)                                    [Loss]⎼> (output)
                                                ╱
                                        (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto ea_input = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {ea_input});

    operation::ElementwiseActivation::Param param;
    param.op_type = operation::ElementwiseActivation::Type::RELU;
    param.alpha = std::numeric_limits<float>::infinity();
    param.beta = 0.f;
    const auto ea_op = operation::ElementwiseActivation({ea_input}, {y_pred}, param);
    const auto ea_index =
      tgraph.addOperation(std::make_unique<train::operation::ElementwiseActivation>(ea_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_ea_index{ea_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_ea_index{ea_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_ea_input_uses{
      forwarding_ea_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_ea_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_ea_input =
      createUseDefChain(tgraph.operands().at(ea_input), expected_forwarding_ea_input_uses,
                        expected_forwarding_ea_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_ea_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{forwarding_ea_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_ea_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_ea_input_def{
      backwarding_ea_index};
    const auto expected_backwarding_ea_input =
      createUseDefChain(tgraph.operands().at(ea_input), expected_backwarding_ea_input_uses,
                        expected_backwarding_ea_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_ea_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_ea_input ==
                training_usedefs.at(train::TrainingOperandIndex{ea_input, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_ea_input ==
                training_usedefs.at(train::TrainingOperandIndex{ea_input, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // FullyConnected
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (y_pred)
             ╱               ╲
    (weights)                 [Loss]⎼> (output)
                             ╱
                     (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {y_pred});
    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{forwarding_fc_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Pad
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> weights_data(4, 0.f);
    std::vector<int32_t> padding_data(4, 0);

    /*
     (input) ⎼[FC]⎼> (pad_input) ⎼[Pad]⎼> (y_pred)
             ╱                   ╱                ╲
    (weights)           (padding)                  [Loss]⎼> (output)
                                                  ╱
                                          (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto pad_input = tgraph.addOperand(shape, type);
    const auto padding = tgraph.addOperand(shape, TypeInfo{DataType::INT32});
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(weights_data.data()), weights_data.size() * sizeof(float)));
    tgraph.operands().at(padding).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(padding_data.data()), padding_data.size() * sizeof(int32_t)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {pad_input});

    const auto pad_op = operation::Pad({pad_input, padding}, {y_pred});
    const auto pad_index = tgraph.addOperation(std::make_unique<train::operation::Pad>(pad_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_pad_index{pad_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_pad_index{pad_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_pad_input_uses{
      forwarding_pad_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_pad_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_pad_input =
      createUseDefChain(tgraph.operands().at(pad_input), expected_forwarding_pad_input_uses,
                        expected_forwarding_pad_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_padding_uses{
      forwarding_pad_index, backwarding_pad_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_padding_def{};
    const auto expected_forwarding_padding =
      createUseDefChain(tgraph.operands().at(padding), expected_forwarding_padding_uses,
                        expected_forwarding_padding_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{forwarding_pad_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_pad_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_pad_input_def{
      backwarding_pad_index};
    const auto expected_backwarding_pad_input =
      createUseDefChain(tgraph.operands().at(pad_input), expected_backwarding_pad_input_uses,
                        expected_backwarding_pad_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_padding_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_padding_def{};
    const auto expected_backwarding_padding =
      createUseDefChain(tgraph.operands().at(padding), expected_backwarding_padding_uses,
                        expected_backwarding_padding_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_pad_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_pad_input ==
                training_usedefs.at(train::TrainingOperandIndex{pad_input, true}));
    EXPECT_TRUE(expected_forwarding_padding ==
                training_usedefs.at(train::TrainingOperandIndex{padding, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_pad_input ==
                training_usedefs.at(train::TrainingOperandIndex{pad_input, false}));
    EXPECT_TRUE(expected_backwarding_padding ==
                training_usedefs.at(train::TrainingOperandIndex{padding, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Pool2D - Max
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (pool_input) ⎼[MaxPool2D]⎼> (y_pred)
             ╱                                           ╲
    (weights)                                             [Loss]⎼> (output)
                                                         ╱
                                                 (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto pool_input = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {pool_input});

    operation::Pool2D::Param param;
    param.op_type = operation::Pool2D::PoolType::MAX;
    param.kh = 0;
    param.kw = 0;
    param.padding = PaddingType::VALID;
    param.stride = Stride{1, 1};
    param.activation = Activation::NONE;
    const auto pool_op = operation::Pool2D({pool_input}, {y_pred}, param);
    const auto pool_index =
      tgraph.addOperation(std::make_unique<train::operation::Pool2D>(pool_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_pool_index{pool_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_pool_index{pool_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_pool_input_uses{
      forwarding_pool_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_pool_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_pool_input =
      createUseDefChain(tgraph.operands().at(pool_input), expected_forwarding_pool_input_uses,
                        expected_forwarding_pool_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_pool_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_pool_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_pool_input_def{
      backwarding_pool_index};
    const auto expected_backwarding_pool_input =
      createUseDefChain(tgraph.operands().at(pool_input), expected_backwarding_pool_input_uses,
                        expected_backwarding_pool_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_pool_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_pool_input ==
                training_usedefs.at(train::TrainingOperandIndex{pool_input, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_pool_input ==
                training_usedefs.at(train::TrainingOperandIndex{pool_input, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Reduce - Mean
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 1};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> weights_data(4, 0.f);
    std::vector<int32_t> axis_data{-1};

    /*
     (input) ⎼[FC]⎼> (mean_input) ⎼[Mean]⎼> (y_pred)
             ╱                    ╱                 ╲
    (weights)               (axis)                   [Loss]⎼> (output)
                                                    ╱
                                            (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto mean_input = tgraph.addOperand(shape, type);
    const auto axis = tgraph.addOperand(Shape{1}, TypeInfo{DataType::INT32});
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(weights_data.data()), weights_data.size() * sizeof(float)));
    tgraph.operands().at(axis).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(axis_data.data()), axis_data.size() * sizeof(int32_t)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {mean_input});

    operation::Reduce::Param param;
    param.reduce_type = operation::Reduce::ReduceType::MEAN;
    param.keep_dims = true;
    const auto mean_op = operation::Reduce({mean_input, axis}, {y_pred}, param);
    const auto mean_index =
      tgraph.addOperation(std::make_unique<train::operation::Reduce>(mean_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_mean_index{mean_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_mean_index{mean_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_mean_input_uses{
      forwarding_mean_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_mean_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_mean_input =
      createUseDefChain(tgraph.operands().at(mean_input), expected_forwarding_mean_input_uses,
                        expected_forwarding_mean_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_axis_uses{forwarding_mean_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_axis_def{};
    const auto expected_forwarding_axis = createUseDefChain(
      tgraph.operands().at(axis), expected_forwarding_axis_uses, expected_forwarding_axis_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_mean_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_mean_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_mean_input_def{
      backwarding_mean_index};
    const auto expected_backwarding_mean_input =
      createUseDefChain(tgraph.operands().at(mean_input), expected_backwarding_mean_input_uses,
                        expected_backwarding_mean_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_axis_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_axis_def{};
    const auto expected_backwarding_axis = createUseDefChain(
      tgraph.operands().at(axis), expected_backwarding_axis_uses, expected_backwarding_axis_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_mean_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_mean_input ==
                training_usedefs.at(train::TrainingOperandIndex{mean_input, true}));
    EXPECT_TRUE(expected_forwarding_axis ==
                training_usedefs.at(train::TrainingOperandIndex{axis, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_mean_input ==
                training_usedefs.at(train::TrainingOperandIndex{mean_input, false}));
    EXPECT_TRUE(expected_backwarding_axis ==
                training_usedefs.at(train::TrainingOperandIndex{axis, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Reshape
  {
    train::TrainableGraph tgraph;

    Shape s{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> weights_data(4, 0.f);
    std::vector<int32_t> shape_data{2, 2};

    /*
     (input) ⎼[FC]⎼> (reshape_input) ⎼[Reshape]⎼> (y_pred)
             ╱                       ╱                    ╲
    (weights)                 (shape)                      [Loss]⎼> (output)
                                                          ╱
                                                  (y_true)
    */

    const auto input = tgraph.addOperand(s, type);
    const auto weights = tgraph.addOperand(s, type);
    const auto reshape_input = tgraph.addOperand(s, type);
    const auto shape = tgraph.addOperand(Shape{2}, TypeInfo{DataType::INT32});
    const auto y_pred = tgraph.addOperand(s, type);
    const auto y_true = tgraph.addOperand(s, type);
    const auto output = tgraph.addOperand(s, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(weights_data.data()), weights_data.size() * sizeof(float)));
    tgraph.operands().at(shape).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(shape_data.data()), shape_data.size() * sizeof(int32_t)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {reshape_input});

    operation::Reshape::Param param;
    param.new_shape = shape_data;
    const auto reshape_op = operation::Reshape({reshape_input, shape}, {y_pred}, param);
    const auto reshape_index =
      tgraph.addOperation(std::make_unique<train::operation::Reshape>(reshape_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_reshape_index{reshape_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_reshape_index{reshape_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_reshape_input_uses{
      forwarding_reshape_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_reshape_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_reshape_input =
      createUseDefChain(tgraph.operands().at(reshape_input), expected_forwarding_reshape_input_uses,
                        expected_forwarding_reshape_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_shape_uses{
      forwarding_reshape_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_shape_def{};
    const auto expected_forwarding_axis = createUseDefChain(
      tgraph.operands().at(shape), expected_forwarding_shape_uses, expected_forwarding_shape_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_reshape_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_reshape_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_reshape_input_def{
      backwarding_reshape_index};
    const auto expected_backwarding_reshape_input = createUseDefChain(
      tgraph.operands().at(reshape_input), expected_backwarding_reshape_input_uses,
      expected_backwarding_reshape_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_shape_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_shape_def{};
    const auto expected_backwarding_axis = createUseDefChain(
      tgraph.operands().at(shape), expected_backwarding_shape_uses, expected_backwarding_shape_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_reshape_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_reshape_input ==
                training_usedefs.at(train::TrainingOperandIndex{reshape_input, true}));
    EXPECT_TRUE(expected_forwarding_axis ==
                training_usedefs.at(train::TrainingOperandIndex{shape, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_reshape_input ==
                training_usedefs.at(train::TrainingOperandIndex{reshape_input, false}));
    EXPECT_TRUE(expected_backwarding_axis ==
                training_usedefs.at(train::TrainingOperandIndex{shape, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Softmax
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (softmax_input) ⎼[Softmax]⎼> (y_pred)
             ╱                                            ╲
    (weights)                                              [Loss]⎼> (output)
                                                          ╱
                                                  (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto softmax_input = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {softmax_input});

    operation::Softmax::Param param;
    param.beta = 1.0f;
    const auto softmax_op = operation::Softmax({softmax_input}, {y_pred}, param);
    const auto softmax_index =
      tgraph.addOperation(std::make_unique<train::operation::Softmax>(softmax_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_softmax_index{softmax_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_softmax_index{softmax_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_softmax_input_uses{
      forwarding_softmax_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_softmax_input_def{
      forwarding_fc_index};
    const auto expected_forwarding_softmax_input =
      createUseDefChain(tgraph.operands().at(softmax_input), expected_forwarding_softmax_input_uses,
                        expected_forwarding_softmax_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_softmax_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{
      forwarding_softmax_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_softmax_input_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_softmax_input_def{
      backwarding_softmax_index};
    const auto expected_backwarding_softmax_input = createUseDefChain(
      tgraph.operands().at(softmax_input), expected_backwarding_softmax_input_uses,
      expected_backwarding_softmax_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_softmax_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_softmax_input ==
                training_usedefs.at(train::TrainingOperandIndex{softmax_input, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_softmax_input ==
                training_usedefs.at(train::TrainingOperandIndex{softmax_input, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }

  // Branch ops
  {
    train::TrainableGraph tgraph;

    Shape shape{2, 2};
    TypeInfo type{DataType::FLOAT32};
    std::vector<float> data(4, 0.f);

    /*
     (input) ⎼[FC]⎼> (fc_out) ⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼⎼[BA]⎼> (y_pred)
             ╱               ╲               ╱               ╲
    (weights)                 [EA]⎼> (ea_out)                 [Loss]⎼> (output)
                                                             ╱
                                                     (y_true)
    */

    const auto input = tgraph.addOperand(shape, type);
    const auto weights = tgraph.addOperand(shape, type);
    const auto fc_out = tgraph.addOperand(shape, type);
    const auto ea_out = tgraph.addOperand(shape, type);
    const auto y_pred = tgraph.addOperand(shape, type);
    const auto y_true = tgraph.addOperand(shape, type);
    const auto output = tgraph.addOperand(shape, type);

    tgraph.operands().at(weights).data(std::make_unique<ExternalData>(
      reinterpret_cast<uint8_t *>(data.data()), data.size() * sizeof(float)));

    tgraph.addInput({input});
    tgraph.addInput({y_true});
    tgraph.addOutput({output});

    const auto fc_index =
      addFullyConnectedOperation(tgraph, {input, weights, OperandIndex{}}, {fc_out});

    operation::ElementwiseActivation::Param ea_param;
    ea_param.op_type = operation::ElementwiseActivation::Type::RELU;
    ea_param.alpha = std::numeric_limits<float>::infinity();
    ea_param.beta = 0.f;
    const auto ea_op = operation::ElementwiseActivation({fc_out}, {ea_out}, ea_param);
    const auto ea_index =
      tgraph.addOperation(std::make_unique<train::operation::ElementwiseActivation>(ea_op));

    operation::BinaryArithmetic::Param ba_param;
    ba_param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;
    ba_param.activation = Activation::NONE;
    const auto ba_op = operation::BinaryArithmetic({fc_out, ea_out}, {y_pred}, ba_param);
    const auto ba_index =
      tgraph.addOperation(std::make_unique<train::operation::BinaryArithmetic>(ba_op));

    const auto loss_index = addLossOperation(tgraph, {y_pred, y_true}, {output});

    enableAllBackwarding(tgraph);

    EXPECT_NO_THROW(tgraph.setTrainingUseDefs(train::UseDefGenerator{tgraph}()));
    EXPECT_NO_THROW(tgraph.verify());

    train::TrainingOperationIndex forwarding_fc_index{fc_index, true};
    train::TrainingOperationIndex forwarding_ea_index{ea_index, true};
    train::TrainingOperationIndex forwarding_ba_index{ba_index, true};
    train::TrainingOperationIndex forwarding_loss_index{loss_index, true};
    train::TrainingOperationIndex backwarding_fc_index{fc_index, false};
    train::TrainingOperationIndex backwarding_ea_index{ea_index, false};
    train::TrainingOperationIndex backwarding_ba_index{ba_index, false};
    train::TrainingOperationIndex backwarding_loss_index{loss_index, false};

    std::vector<train::TrainingOperationIndex> expected_forwarding_input_uses{forwarding_fc_index,
                                                                              backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_input_def{};
    const auto expected_forwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_forwarding_input_uses, expected_forwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_uses{
      forwarding_fc_index, backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_weights_def{};
    const auto expected_forwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_forwarding_weights_uses,
                        expected_forwarding_weights_def);

    // forwarding output of FC is not used in backwarding if activation is none
    std::vector<train::TrainingOperationIndex> expected_forwarding_fc_out_uses{
      forwarding_ea_index, forwarding_ba_index, backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_fc_out_def{forwarding_fc_index};
    const auto expected_forwarding_fc_out =
      createUseDefChain(tgraph.operands().at(fc_out), expected_forwarding_fc_out_uses,
                        expected_forwarding_fc_out_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_ea_out_uses{
      forwarding_ba_index, backwarding_ea_index, backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_ea_out_def{forwarding_ea_index};
    const auto expected_forwarding_ea_out =
      createUseDefChain(tgraph.operands().at(ea_out), expected_forwarding_ea_out_uses,
                        expected_forwarding_ea_out_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_pred_def{forwarding_ba_index};
    const auto expected_forwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_forwarding_y_pred_uses,
                        expected_forwarding_y_pred_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_uses{
      forwarding_loss_index, backwarding_loss_index};
    std::vector<train::TrainingOperationIndex> expected_forwarding_y_true_def{};
    const auto expected_forwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_forwarding_y_true_uses,
                        expected_forwarding_y_true_def);

    std::vector<train::TrainingOperationIndex> expected_forwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_forwarding_output_def{
      forwarding_loss_index};
    const auto expected_forwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_forwarding_output_uses,
                        expected_forwarding_output_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_input_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_input_def{backwarding_fc_index};
    const auto expected_backwarding_input = createUseDefChain(
      tgraph.operands().at(input), expected_backwarding_input_uses, expected_backwarding_input_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_weights_def{
      backwarding_fc_index};
    const auto expected_backwarding_weights =
      createUseDefChain(tgraph.operands().at(weights), expected_backwarding_weights_uses,
                        expected_backwarding_weights_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_fc_out_uses{
      backwarding_fc_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_fc_out_def{
      backwarding_ea_index, backwarding_ba_index};
    const auto expected_backwarding_fc_out =
      createUseDefChain(tgraph.operands().at(fc_out), expected_backwarding_fc_out_uses,
                        expected_backwarding_fc_out_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_ea_out_uses{
      backwarding_ea_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_ea_out_def{
      backwarding_ba_index};
    const auto expected_backwarding_ea_out =
      createUseDefChain(tgraph.operands().at(ea_out), expected_backwarding_ea_out_uses,
                        expected_backwarding_ea_out_def);

    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_uses{
      backwarding_ba_index};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_pred_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_pred =
      createUseDefChain(tgraph.operands().at(y_pred), expected_backwarding_y_pred_uses,
                        expected_backwarding_y_pred_def);

    // backwarding_y_true is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_y_true_def{
      backwarding_loss_index};
    const auto expected_backwarding_y_true =
      createUseDefChain(tgraph.operands().at(y_true), expected_backwarding_y_true_uses,
                        expected_backwarding_y_true_def);

    // backwarding_output is not defined and not used by any operation
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_uses{};
    std::vector<train::TrainingOperationIndex> expected_backwarding_output_def{};
    const auto expected_backwarding_output =
      createUseDefChain(tgraph.operands().at(output), expected_backwarding_output_uses,
                        expected_backwarding_output_def);

    const auto &training_usedefs = tgraph.trainingUseDefs();

    EXPECT_TRUE(expected_forwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, true}));
    EXPECT_TRUE(expected_forwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, true}));
    EXPECT_TRUE(expected_forwarding_fc_out ==
                training_usedefs.at(train::TrainingOperandIndex{fc_out, true}));
    EXPECT_TRUE(expected_forwarding_ea_out ==
                training_usedefs.at(train::TrainingOperandIndex{ea_out, true}));
    EXPECT_TRUE(expected_forwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, true}));
    EXPECT_TRUE(expected_forwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, true}));
    EXPECT_TRUE(expected_forwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, true}));

    EXPECT_TRUE(expected_backwarding_input ==
                training_usedefs.at(train::TrainingOperandIndex{input, false}));
    EXPECT_TRUE(expected_backwarding_weights ==
                training_usedefs.at(train::TrainingOperandIndex{weights, false}));
    EXPECT_TRUE(expected_backwarding_fc_out ==
                training_usedefs.at(train::TrainingOperandIndex{fc_out, false}));
    EXPECT_TRUE(expected_backwarding_ea_out ==
                training_usedefs.at(train::TrainingOperandIndex{ea_out, false}));
    EXPECT_TRUE(expected_backwarding_y_pred ==
                training_usedefs.at(train::TrainingOperandIndex{y_pred, false}));
    EXPECT_TRUE(expected_backwarding_y_true ==
                training_usedefs.at(train::TrainingOperandIndex{y_true, false}));
    EXPECT_TRUE(expected_backwarding_output ==
                training_usedefs.at(train::TrainingOperandIndex{output, false}));
  }
}
