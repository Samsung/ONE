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

TEST_F(GenModelTrain, OneOp_Conv2D)
{
  CirclePlusGen cgen;

  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);

  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  float learning_rate = 1.0f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, 0.01f,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  // NOTE loss value
  // The innermost curly braces contain the loss values for each output index,
  // and the other curly braces contain the loss values by epoch.
  // example)
  //   Total output number: 2
  //   Epoch: 3
  //   ------------------------
  //   Loss: { {x,x}, {x,x}, {x,x} }
  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(uniformTCD<float>(
    {{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
      1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}},                               // inputs
    {{97, -4, -25, 9, 90, 10, -13, 91, -14, -26, -12, 26, 20, 90, 1, 3, 11, 4}}, // expected
    {{988.8889f}, {818.4783f}, {702.7875f}}                                      // loss
    ));
  _context->setBackends({"train"});
  _context->setEpoch(3);

  SUCCEED();
}
