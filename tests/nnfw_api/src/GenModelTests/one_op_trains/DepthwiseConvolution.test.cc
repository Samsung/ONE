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

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 2 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{1, 2, 7, 8, 3, 4, 9, 1, 5, 6, 11, 2}}}, // input dataset
                      {{{1, -4, 1, -3, 2, -2, 2, -4}}},          // expected dataset
                      {13.107666f}                               // last losses
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(2);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D_No_Multiplier)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(3 * 2, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 1, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}}}, // input dataset
                      {{{6.5f, 7.5f, 8.5f, 3.5f, 8.5f, 5.5f, 2.5f, 3.5f}}}, // expected dataset
                      {74.799194f}                                          // last losses
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(2);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D_No_Multiplier_RELU6)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(3 * 2, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 1, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_RELU6);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}}}, // input dataset
                      {{{6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f}}}, // expected dataset
                      {72.f}                                                // last losses
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(2);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_DepthwiseConv2D_3x3)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(3 * 3 * 2, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(uniformTCD<float>(
    {{{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}}},       // input dataset
    {{{6.0f, 16.0f, 8.0f, 16.0f, 10.0f, 16.0f, 12.0f, 16.0f}}}, // expected dataset
    {329.703643f}                                               // last losses
    ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(2);

  SUCCEED();
}

// TODO Add tests for dilation

TEST_F(GenModelTrain, neg_OneOp_DepthwiseConv2D_Stride)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 2 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 0, 0, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTrain, neg_OneOp_DepthwiseConv2D_Dilation)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 2 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));
  int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE, 0, 0);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTrain, neg_OneOp_DepthwiseConv2D_Type)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 2 * 4, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(4, 0.f));
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_UINT8});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 0.001f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailModelLoad();

  SUCCEED();
}
