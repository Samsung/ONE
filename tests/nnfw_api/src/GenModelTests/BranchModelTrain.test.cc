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

TEST_F(GenModelTrain, BranchOps_FC_BinaryArithmetic)
{
  // (( Input 0 )) -> [ FC ] ----\
  //                              |=> [ Add ] -> (( Output 0 ))
  // (( Input 1 )) --------------/
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
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(uniformTCD<float>(
      {{{1, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{2, 1}, {7, 6, 5, 4, 3, 2, 1, 0}}}, // inputs
      {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}},                 // expected
      {8.4678f}                                                                 // loss
      ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }

  // (( Input 0 )) --------------\
  //                              |=> [ Sub ] -> (( Output 0 ))
  // (( Input 1 )) -> [ FC ] ----/
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
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(uniformTCD<float>(
      {{{0, 1, 2, 3, 4, 5, 1, 3}, {6, 7}}, {{5, 4, 3, 2, 1, 0, 2, 1}, {7, 6}}}, // inputs
      {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}},                 // expected
      {3.2863f}                                                                 // loss
      ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }

  // (( Input 0 )) -> [ FC ] ----\
  //                              |=> [ Mul ] -> (( Output 0 ))
  // (( Input 1 )) -> [ FC ] ----/
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
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{0, 3}, {6, 7}}, {{5, 4}, {7, 6}}},                     // inputs
                        {{{3, 2, 1, 2, 5, 6, 1, 0}}, {{2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                        {12.2822f}                                                // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}
