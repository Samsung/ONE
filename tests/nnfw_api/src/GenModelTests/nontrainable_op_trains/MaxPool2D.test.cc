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

auto getDefaultPool2DOptions()
{
  return std::tuple<circle::Padding, int, int, int, int, circle::ActivationFunctionType>(
    circle::Padding_SAME, 1, 1, 1, 1, circle::ActivationFunctionType_NONE);
}

} // namespace

/*
Epoch 1/5
1/1 [==============================] - 0s 152ms/step - loss: 31.6667 - categorical_accuracy: 0.5556
Epoch 2/5
1/1 [==============================] - 0s 2ms/step - loss: 28.6837 - categorical_accuracy: 0.6667
Epoch 3/5
1/1 [==============================] - 0s 2ms/step - loss: 26.1765 - categorical_accuracy: 0.7778
Epoch 4/5
1/1 [==============================] - 0s 2ms/step - loss: 24.0089 - categorical_accuracy: 0.7778
Epoch 5/5
1/1 [==============================] - 0s 2ms/step - loss: 22.0964 - categorical_accuracy: 0.8889
*/
TEST_F(GenModelTrain, NonTrainableOps_Conv2D_MaxPool2D)
{
  // (( Input 0 )) -> [ Conv2D ] -> [ MaxPool2D ] -> (( Output 0 ))
  // Padding : Same
  // stride height : 1, stride width : 1
  // filter height : 1, filter hight : 1
  // Activation : None
  {
    CirclePlusGen cgen;
    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 3 * 3, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
    int input = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int conv_output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
    const auto [padding, stride_w, stride_h, filter_w, filter_h, actfn] = getDefaultPool2DOptions();

    cgen.addOperatorConv2D({{input, weight, bias}, {conv_output}}, circle::Padding_VALID, 1, 1,
                           circle::ActivationFunctionType_NONE, 1, 1);
    cgen.addOperatorMaxPool2D({{conv_output}, {output}}, padding, stride_w, stride_h, filter_w,
                              filter_h, actfn);
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
                           1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}}},            // inputs
                        {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1}}}, // expected
                        {24.0089f}                                                  // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(4);

    SUCCEED();
  }
}

/*
Epoch 1/5
1/1 [==============================] - 0s 150ms/step - loss: 31.6667 - categorical_accuracy: 0.5556
Epoch 2/5
1/1 [==============================] - 0s 2ms/step - loss: 27.8823 - categorical_accuracy: 0.5556
Epoch 3/5
1/1 [==============================] - 0s 2ms/step - loss: 16.9743 - categorical_accuracy: 0.6667
Epoch 4/5
1/1 [==============================] - 0s 2ms/step - loss: 9.3556 - categorical_accuracy: 0.6667
Epoch 5/5
1/1 [==============================] - 0s 2ms/step - loss: 5.8163 - categorical_accuracy: 0.6667
*/
TEST_F(GenModelTrain, NonTrainableOps_Conv2D_MaxPool2D_Filter2)
{
  // (( Input 0 )) -> [ Conv2D ] -> [ MaxPool2D ] -> (( Output 0 ))
  // Padding : Same
  // stride height : 1, stride width : 1
  // filter height : 2, filter hight : 2
  // Activation : None
  {
    CirclePlusGen cgen;
    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 3 * 3, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
    int input = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int conv_output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
    auto [padding, stride_w, stride_h, filter_w, filter_h, actfn] = getDefaultPool2DOptions();
    // padding = circle::Padding_VALID;
    filter_w = 2;
    filter_h = 2;

    cgen.addOperatorConv2D({{input, weight, bias}, {conv_output}}, circle::Padding_VALID, 1, 1,
                           circle::ActivationFunctionType_NONE, 1, 1);
    cgen.addOperatorMaxPool2D({{conv_output}, {output}}, padding, stride_w, stride_h, filter_w,
                              filter_h, actfn);
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
                           1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}}},            // inputs
                        {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1}}}, // expected
                        {9.3556f}                                                   // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(2);

    SUCCEED();
  }
}

/*
Epoch 1/5
1/1 [==============================] - 0s 150ms/step - loss: 25.5000 - categorical_accuracy:
0.0000e+00 Epoch 2/5 1/1 [==============================] - 0s 2ms/step - loss: 19.2126 -
categorical_accuracy: 0.7500 Epoch 3/5 1/1 [==============================] - 0s 2ms/step -
loss: 12.9202 - categorical_accuracy: 1.0000 Epoch 4/5 1/1 [==============================] - 0s
2ms/step - loss: 9.0784 - categorical_accuracy: 1.0000 Epoch 5/5 1/1
[==============================] - 0s 2ms/step - loss: 6.6907 - categorical_accuracy: 1.0000
*/
TEST_F(GenModelTrain, NonTrainableOps_Conv2D_MaxPool2D_Stride2Filter2)
{
  // (( Input 0 )) -> [ Conv2D ] -> [ MaxPool2D ] -> (( Output 0 ))
  // Padding : Same
  // stride height : 2, stride width : 2
  // filter height : 2, filter hight : 2
  // Activation : None
  {
    CirclePlusGen cgen;
    uint32_t weight_buf = cgen.addBuffer(std::vector<float>(2 * 3 * 3, 0.f));
    uint32_t bias_buf = cgen.addBuffer(std::vector<float>(2, 0.f));
    int input = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
    int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
    int bias = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
    int conv_output = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
    int output = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
    auto [padding, stride_w, stride_h, filter_w, filter_h, actfn] = getDefaultPool2DOptions();
    stride_w = 2;
    stride_h = 2;
    filter_w = 2;
    filter_h = 2;

    cgen.addOperatorConv2D({{input, weight, bias}, {conv_output}}, circle::Padding_VALID, 1, 1,
                           circle::ActivationFunctionType_NONE, 1, 1);
    cgen.addOperatorMaxPool2D({{conv_output}, {output}}, padding, stride_w, stride_h, filter_w,
                              filter_h, actfn);
    cgen.setInputsAndOutputs({input}, {output});

    float learning_rate = 0.01f;
    int32_t batch_size = 1;
    cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                       circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                       circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

    _context = std::make_unique<GenModelTrainContext>(cgen.finish());
    _context->addTrainCase(
      uniformTCD<float>({{{4, 0,  -5, 1, 0,  4, -1, 1, -1, -3, 3,  -2, -4,
                           1, -2, 2,  4, -4, 2, 2,  0, 4,  -1, -2, 4}}}, // inputs
                        {{{1, 2, 3, 4, 5, 6, 7, 8}}},                    // expected
                        {9.0784f}                                        // loss
                        ));

    _context->setBackends({"train"});
    _context->setEpoch(2);

    SUCCEED();
  }
}

TEST_F(GenModelTrain, neg_NonTrainableOps_MaxPool2D_InvalidShape)
{
  CirclePlusGen cgen;
  int input = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int output = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  const auto [padding, stride_w, stride_h, filter_w, filter_h, actfn] = getDefaultPool2DOptions();

  cgen.addOperatorMaxPool2D({{input}, {output}}, padding, stride_w, stride_h, filter_w, filter_h,
                            actfn);
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

TEST_F(GenModelTrain, neg_NonTrainableOps_MaxPool2D_InvalidType)
{
  CirclePlusGen cgen;
  int input = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  const auto [padding, stride_w, stride_h, filter_w, filter_h, actfn] = getDefaultPool2DOptions();
  cgen.addOperatorMaxPool2D({{input}, {output}}, padding, stride_w, stride_h, filter_w, filter_h,
                            actfn);
  cgen.setInputsAndOutputs({input, padding}, {output});

  float learning_rate = 0.01f;
  int32_t batch_size = 1;
  cgen.addTrainInfo({circle::Optimizer::Optimizer_SGD, learning_rate,
                     circle::LossFn::LossFn_MEAN_SQUARED_ERROR,
                     circle::LossReductionType::LossReductionType_SumOverBatchSize, batch_size});

  _context = std::make_unique<GenModelTrainContext>(cgen.finish());
  _context->setBackends({"train"});
  _context->expectFailCompile();
}
