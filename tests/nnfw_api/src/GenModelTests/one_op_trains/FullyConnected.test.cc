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

#include <memory>

TEST_F(GenModelTrain, OneOp_FullyConnected)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(8, 0.f));
  int input = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD,
                     learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize,
                     batch_size,
                     {0}});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{1, 3}}, {{2, 1}}},                                     // inputs
                      {{{2, 1, 5, 5, 2, 1, 5, 5}}, {{2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                      {11.4484f}                                                // loss
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(4);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_FullyConnected_OptionalBias)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(8 * 2, 0.f));
  int input = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{8, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int output = cgen.addTensor({{1, 8}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, -1 /* Optional bias */}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 2;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD,
                     learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize,
                     batch_size,
                     {0}});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{1, 3, 2, 1}}},                                     // inputs
                      {{{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}}}, // expected
                      {{12.7512f}}                                          // loss
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(5);

  SUCCEED();
}

TEST_F(GenModelTrain, neg_OneOp_FullyConnected_FourOperand)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(16 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(16, 0.f));
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int wrong = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias, wrong}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD,
                     learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize,
                     batch_size,
                     {0}});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTrain, neg_OneOp_FullyConnected_InvalidWeightShape)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(16 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(16, 0.f));
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{15, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int wrong = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias, wrong}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD,
                     learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize,
                     batch_size,
                     {0}});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTrain, neg_OneOp_FullyConnected_NoBias)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(16 * 4, 0.f));
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight /* Missing bias */}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD,
                     learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize,
                     batch_size,
                     {0}});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailCompile();

  SUCCEED();
}
