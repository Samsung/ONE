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

namespace
{
CircleBuffers get_conv2d_test_model()
{
  CirclePlusGen cgen;

  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 3 * 3, 0.f));
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, 1.0f,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});
  return cgen.finish();
}
} // namespace

TEST_F(GenModelTrain, OneOp_Conv2D)
{
  _context = std::make_unique<GenModelTrainContext>(get_conv2d_test_model());
  _context->addTrainCase(uniformTCD<float>(
    {{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
      1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}},                               // inputs
    {{47, -4, -25, 9, 10, 10, -13, 11, -14, -26, -12, 26, 20, 40, 1, 3, 11, 4}}, // expected
    {{62469.609375f}}                                                            // loss
    ));
  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(2);

  SUCCEED();
}

TEST_F(GenModelTrain, OneOp_Conv2D_disable_update)
{
  _context = std::make_unique<GenModelTrainContext>(get_conv2d_test_model());
  _context->addTrainCase(uniformTCD<float>(
    {{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
      1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}},                               // inputs
    {{47, -4, -25, 9, 10, 10, -13, 11, -14, -26, -12, 26, 20, 40, 1, 3, 11, 4}}, // expected
    {{806.667}}                                                                  // loss
    ));
  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->disableTrainNodeUpdate(0); // the test model contains only one op
  _context->setEpoch(2);               // gain of loss after each epoch is const (equal 403.333)

  SUCCEED();
}
