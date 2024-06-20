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

TEST_F(GenModelTrain, NonTrainableOps_FC_Pad)
{
  // (( Input 0 )) -> [ FC ] -> [ Pad ] -> (( Output 0 ))
  {
    CirclePlusGen cgen;
    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
    uint32_t padding_buf = cgen.addBuffer(std::vector<int32_t>{0, 0, 1, 1});
    int input = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{2, 8}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int fc_output = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
    int padding = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32, padding_buf});
    int output = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorFullyConnected({{input, weight, bias}, {fc_output}});
    cgen.addOperatorPad({{fc_output, padding}, {output}});
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});
    cgen.markAllOpsAsTrainable();

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{0, 1, 2, 3, 4, 5, 6, 7}}, {{7, 6, 5, 4, 3, 2, 1, 0}}}, // inputs
                        {{{0, 13, 52, 0}}, {{0, 31, 24, 0}}},                     // expected
                        {1.3900f}                                                 // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Pad_InvalidShape)
{
  CirclePlusGen cgen;
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int input = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  // Invalid shape: output shape should be equal to input shape
  int output = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorPad({{input, padding}, {output}});
  cgen.setInputsAndOutputs({input, padding}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});
  cgen.markAllOpsAsTrainable();

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailCompile();
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Pad_InvalidType)
{
  CirclePlusGen cgen;
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  // Invalid  type: input tensor type should be FLOAT32
  int input = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int output = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorPad({{input, padding}, {output}});
  cgen.setInputsAndOutputs({input, padding}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});
  cgen.markAllOpsAsTrainable();

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();
}
