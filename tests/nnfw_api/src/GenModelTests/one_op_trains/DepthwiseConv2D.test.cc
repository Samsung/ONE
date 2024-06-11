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

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2*2*4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));

  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(uniformTCD<float>({{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}},
                                          {{71, -34, 99, -20, 91, -26, 127, -4}},
                                          {{5187.5000}, {176.7085}, {42.2953}}));
  _context->setBackends({"train"});
  _context->setEpoch(3);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D_No_Multiplier)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(3*1*2, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));

  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 1, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
                      {{16.5f, 27.5f, 28.5f, 43.5f, 8.5f, 15.5f, 12.5f, 23.5f}},
                      {{594.2500}, {341.8356}, {201.5387}}));
  _context->setBackends({"train"});
  _context->setEpoch(3);

  SUCCEED();
}

// TEST_F(GenModelTrain, OneOp_DepthwiseConv2D_Dilation)
// {
//   CirclePlusGen cgen;

//   uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2*2*4, 0.f));
//   uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));

//   int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
//   int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
//   int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
//   int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
//   cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
//                                   circle::ActivationFunctionType_NONE, 2, 2);
//   cgen.setInputsAndOutputs({in}, {out});

//   float learning_rate = 0.01f;
//   int32_t batch_size = 1;
//   cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
//                      circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
//                      circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

//   _context = std::make_unique<GenModelTrainContext>(cgen.finish());
//   _context->addTrainCase(uniformTCD<float>({{
//                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
//                                             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                                           }},
//                                           {{13, 14, 0, 0, 0, 0, 11, 12, 5, 6, 0, 0, 0, 0, 3, 4}},
//                                           {{44.7500}, {44.4551}, {44.1633}}));
//   _context->setBackends({"train"});
//   _context->setEpoch(3);

//   SUCCEED();
// }
