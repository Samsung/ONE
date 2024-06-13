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

/*
Epoch 1/5
1/1 [==============================] - 0s 163ms/step - loss: 403.3333 - categorical_accuracy: 1.0000
Epoch 2/5
1/1 [==============================] - 0s 2ms/step - loss: 324.0978 - categorical_accuracy:
0.0000e+00 Epoch 3/5 1/1 [==============================] - 0s 2ms/step - loss: 267.7882 -
categorical_accuracy: 0.0000e+00 Epoch 4/5 1/1 [==============================] - 0s 2ms/step -
loss: 226.5260 - categorical_accuracy: 0.0000e+00 Epoch 5/5 1/1 [==============================] -
0s 2ms/step - loss: 195.3313 - categorical_accuracy: 0.0000e+00
*/
TEST_F(GenModelTrain, NonTrainableOps_FC_Reshape)
{
  CirclePlusGen cgen;

  uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 3 * 3, 0.f));
  uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
  const auto new_shape = CircleGen::Shape{1, 18};
  uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>(new_shape));
  int input = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int conv_output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int output = cgen.addTensor({{1, 18}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{input, weight, bias}, {conv_output}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.addOperatorReshape({{conv_output, shape}, {output}}, &new_shape);
  cgen.setInputsAndOutputs({input}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->addTrainCase(
    uniformTCD<float>({{{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
                         1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}}}, // input dataset
                      {{{47, -4, -25, 9, 10, 10, -13, 11, -14, -26, -12, 26, 20, 40, 1, 3, 11,
                         4}}},    // expected dataset
                      {226.5260f} // last losses
                      ));

  _context->setBackends({"train"});
  // To apply backward to loss, epoch should be >= 2
  _context->setEpoch(4);

  SUCCEED();
}

TEST_F(GenModelTrain, neg_NonTrainableOps_Reshape_InvalidShape)
{
  CirclePlusGen cgen;

  uint32_t shape_buf = cgen.addBuffer(std::vector<float>{2, 3});
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int output = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorReshape({{input, shape}, {output}});
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

TEST_F(GenModelTrain, neg_NonTrainableOps_Reshape_InvalidType)
{
  CirclePlusGen cgen;

  uint32_t shape_buf = cgen.addBuffer(std::vector<float>{2, 2});
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_INT32});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorReshape({{input, shape}, {output}});
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
