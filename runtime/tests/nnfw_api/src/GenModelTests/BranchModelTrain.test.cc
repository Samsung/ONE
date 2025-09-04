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

#include "GenModelTrain.h"

TEST_F(GenModelTrain, BranchOps_FC_Add)
{
  /*
   * (( Input 0 )) -> [ FC ] ----\
   *                              |=> [ Add ] -> (( Output 0 ))
   * (( Input 1 )) --------------/
   */
  {
    CirclePlusGen cgen;

    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input0 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int input1 = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input0, weight, bias}, {fc_output}});
    cgen.addOperatorAdd({{fc_output, input1}, {output}},
                        circle::ActivationFunctionType::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({input0, input1}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(uniformTCD<float>(
      {{{1, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{2, 1}, {7, 6, 5, 4, 3, 2, 1, 0}}}, // inputs
      {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}},                 // expected
      {{9.2218f}, {8.9554f}, {8.7044f}, {8.4678f}}                              // loss
      ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }

  // (( Input 0 )) -> [ FC ] -> (fc_out) --------------------------╲
  //                                    ╲                           |=> [ Add ] -> (( Output 0 ))
  //                                     ╲-> [ Relu6 ]⎼> (ea_out) -╱
  {
    CirclePlusGen cgen;

    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input0 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int ea_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input0, weight, bias}, {fc_output}});
    cgen.addOperatorRelu6({{fc_output}, {ea_output}});
    cgen.addOperatorAdd({{fc_output, ea_output}, {output}},
                        circle::ActivationFunctionType::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({input0}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{1, 3}}, {{2, 1}}},                                     // inputs
                        {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                        // TODO Modify loss values to results of tensorflow
                        {{14.0124f}, {11.0036f}, {8.1681f}, {6.0974f}} // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, BranchOps_FC_Sub)
{
  /*
   * (( Input 0 )) --------------\
   *                              |=> [ Sub ] -> (( Output 0 ))
   * (( Input 1 )) -> [ FC ] ----/
   */
  {
    CirclePlusGen cgen;

    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input0 = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int input1 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input1, weight, bias}, {fc_output}});
    cgen.addOperatorSub({{input0, fc_output}, {output}},
                        circle::ActivationFunctionType::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({input0, input1}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(uniformTCD<float>(
      {{{0, 1, 2, 3, 4, 5, 1, 3}, {6, 7}}, {{5, 4, 3, 2, 1, 0, 2, 1}, {7, 6}}}, // inputs
      {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}},                 // expected
      {{7.3265f}, {4.6811f}, {3.6735f}, {3.2863f}}                              // loss
      ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }

  // (( Input 0 )) -> [ FC ] -> (fc1_out) ------------------------╲
  //                                     ╲                         |=> [ Sub ] -> (( Output 0 ))
  //                                      ╲-> [ FC ]⎼> (fc2_out) -╱
  {
    CirclePlusGen cgen;

    uint32_t weight1_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias1_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    uint32_t weight2_buf = cgen.addBuffer(std::vector<float>(8 * 8, 0.f));
    uint32_t bias2_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input0 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight1 = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight1_buf});
    int bias1 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias1_buf});
    int fc1_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int weight2 = cgen.addTensor({{8, 8}, circle::TensorType::TensorType_FLOAT32, weight2_buf});
    int bias2 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias2_buf});
    int fc2_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input0, weight1, bias1}, {fc1_output}});
    cgen.addOperatorFullyConnected({{fc1_output, weight2, bias2}, {fc2_output}});
    cgen.addOperatorSub({{fc1_output, fc2_output}, {output}},
                        circle::ActivationFunctionType::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({input0}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{6, 7}}, {{7, 6}}},                                     // inputs
                        {{{2, 1, 5, 7, 2, 5, 5, 5}}, {{2, 1, 5, 2, 2, 1, 3, 6}}}, // expected
                        // TODO Modify loss values to results of tensorflow
                        {{12.9477f}, {5.79475f}, {3.0031f}, {2.3388f}} // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, BranchOps_FC_Mul)
{
  /*
   * (( Input 0 )) -> [ FC ] ----\
   *                              |=> [ Mul ] -> [ FC ] -> (( Output 0 ))
   * (( Input 1 )) -> [ FC ] ----/
   */
  {
    CirclePlusGen cgen;

    uint32_t weight0_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias0_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    uint32_t weight1_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias1_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    uint32_t weight2_buf = cgen.addBuffer(std::vector<float>(8 * 8, 0.f));
    uint32_t bias2_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input0 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int input1 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight0 = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight0_buf});
    int weight1 = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight1_buf});
    int weight2 = cgen.addTensor({{8, 8}, circle::TensorType::TensorType_FLOAT32, weight2_buf});
    int bias0 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias0_buf});
    int bias1 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias1_buf});
    int bias2 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias2_buf});
    int fc_output0 = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int fc_output1 = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int mul_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input0, weight0, bias0}, {fc_output0}});
    cgen.addOperatorFullyConnected({{input1, weight1, bias1}, {fc_output1}});
    cgen.addOperatorMul({{fc_output0, fc_output1}, {mul_output}},
                        circle::ActivationFunctionType::ActivationFunctionType_NONE);
    cgen.addOperatorFullyConnected({{mul_output, weight2, bias2}, {output}});
    cgen.setInputsAndOutputs({input0, input1}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{0, 3}, {6, 7}}, {{5, 4}, {7, 6}}},                     // inputs
                        {{{3, 2, 1, 2, 5, 6, 1, 0}}, {{2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                        {{12.5488f}, {12.4590f}, {12.3701f}, {12.2822f}}          // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }

  // (( Input 0 )) -> [ FC ] -> (fc1_out) ------------------------╲
  //                                     ╲                         |=> [ Mul ] -> (( Output 0 ))
  //                                      ╲-> [ FC ]⎼> (fc2_out) -╱
  {
    CirclePlusGen cgen;

    uint32_t weight1_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias1_buf = cgen.addBuffer(std::vector<float>(8, 1.f));
    uint32_t weight2_buf = cgen.addBuffer(std::vector<float>(8 * 8, 0.f));
    uint32_t bias2_buf = cgen.addBuffer(std::vector<float>(8, 1.f));
    int input0 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight1 = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight1_buf});
    int bias1 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias1_buf});
    int fc1_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int weight2 = cgen.addTensor({{8, 8}, circle::TensorType::TensorType_FLOAT32, weight2_buf});
    int bias2 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias2_buf});
    int fc2_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input0, weight1, bias1}, {fc1_output}});
    cgen.addOperatorFullyConnected({{fc1_output, weight2, bias2}, {fc2_output}});
    cgen.addOperatorMul({{fc1_output, fc2_output}, {output}},
                        circle::ActivationFunctionType::ActivationFunctionType_RELU6);
    cgen.setInputsAndOutputs({input0}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{2, 3}}, {{5, 4}}},                                     // inputs
                        {{{3, 2, 1, 2, 5, 6, 1, 3}}, {{2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                        // TODO Modify loss values to results of tensorflow
                        {{7.6864f}, {5.6564f}, {3.7944f}, {3.1894f}} // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}
