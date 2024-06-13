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

TEST_F(GenModelTrain, NonTrainableOps_FC_Relu)
{
  // (( Input 0 )) -> [ FC ] -> [ Relu ] -> (( Output 0 ))
  {
    CirclePlusGen cgen;

    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
    int input = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input, weight, bias}, {fc_output}});
    cgen.addOperatorRelu({{fc_output}, {output}});
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{1, 3}}, {{2, 1}}},                                     // inputs
                        {{{0, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 0, 1, 5, 6}}}, // expected
                        {13.9375f}                                                // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Relu_InvalidShape)
{
  CirclePlusGen cgen;

  int input = cgen.addTensor({{2, 1}, circle::TensorType::TensorType_FLOAT32});
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorRelu({{input}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailCompile();
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Relu_InvalidType)
{
  CirclePlusGen cgen;

  int input = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorRelu({{input}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();
}
