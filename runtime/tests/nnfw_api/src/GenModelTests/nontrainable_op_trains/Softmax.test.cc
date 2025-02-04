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

TEST_F(GenModelTrain, NonTrainableOps_FC_Softmax)
{
  // (( Input 0 )) -> [ FC ] -> [ Softmax ] -> (( Output 0 ))
  {
    CirclePlusGen cgen;

    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    const float beta = 1.0f;
    cgen.addOperatorFullyConnected({{input, weight, bias}, {fc_output}});
    cgen.addOperatorSoftmax({{fc_output}, {output}}, beta);
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                       NNFW_TRAIN_TRAINABLE_ALL});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{1, 3}}, {{2, 1}}},                                     // inputs
                        {{{0, 1, 0, 0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0, 1, 0, 0}}}, // expected
                        {{0.1094f}, {0.1093f}, {0.1092f}, {0.1092f}}              // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Softmax_InvalidShape)
{
  CirclePlusGen cgen;

  int input = cgen.addTensor({{2, 1}, circle::TensorType::TensorType_FLOAT32});
  // Invalid shape: output shape should be equal to input shape
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  const float beta = 1.0f;
  cgen.addOperatorSoftmax({{input}, {output}}, beta);
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                     NNFW_TRAIN_TRAINABLE_ALL});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Softmax_InvalidType)
{
  CirclePlusGen cgen;

  // Invalid type: input tensor type should be FLOAT32
  int input = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  const float beta = 1.0f;
  cgen.addOperatorSoftmax({{input}, {output}}, beta);
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size,
                     NNFW_TRAIN_TRAINABLE_ALL});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}
