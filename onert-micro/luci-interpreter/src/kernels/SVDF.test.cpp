/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/SVDF.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class SVDFTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(SVDFTest, FullIntegerTest)
{
  const int32_t batches = 2;
  const int32_t input_size = 3;
  const int32_t units = 4;
  const int32_t memory_size = 10;
  const int32_t rank = 1;
  const int32_t num_filters = units * rank;

  Shape input_shape{batches, input_size};
  Shape weight_feature_shape{num_filters, input_size};
  Shape weight_time_shape{num_filters, memory_size};
  Shape bias_shape{units};
  Shape activation_state_shape{batches, memory_size * num_filters};

  std::vector<float> input_data{0.49837467, 0.19278903, 0.26584083,
                                0.17660543, 0.52949083, -0.77931279};

  std::vector<float> weight_feature_data{-0.31930989, -0.36118156, 0.0079667,   0.37613347,
                                         0.22197971,  0.12416199,  0.27901134,  0.27557442,
                                         0.3905206,   -0.36137494, -0.06634006, -0.10640851};

  std::vector<float> weight_time_data{
    -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
    0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

    0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
    -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

    -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
    0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

    -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
    -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

  std::vector<float> bias_data{-0.0976817, 0.15294972, 0.39635518, -0.02702999};

  std::pair<float, int32_t> input_quant_param = quantizationParams<int8_t>(-1, 1);
  std::pair<float, int32_t> weight_feature_quant_param = quantizationParams<int8_t>(-0.5, 0.5);
  std::pair<float, int32_t> weight_time_quant_param = quantizationParams<int16_t>(-1, 1);
  std::pair<float, int32_t> bias_quant_param = quantizationParams<int32_t>(-512, 512);
  std::pair<float, int32_t> activation_state_quant_param = quantizationParams<int16_t>(-16, 16);

  std::pair<float, int32_t> output_quant_param = quantizationParams<int8_t>(-0.5, 0.5);

  Tensor input_tensor =
    makeInputTensor<DataType::S8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor weight_feature_tensor = makeInputTensor<DataType::S8>(
    weight_feature_shape, weight_feature_quant_param.first, weight_feature_quant_param.second,
    weight_feature_data, _memory_manager.get());
  Tensor weight_time_tensor = makeInputTensor<DataType::S16>(
    weight_time_shape, weight_time_quant_param.first, weight_time_quant_param.second,
    weight_time_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(
    bias_shape, bias_quant_param.first, bias_quant_param.second, bias_data, _memory_manager.get());
  Tensor activation_state_tensor = makeOutputTensor(
    DataType::S16, activation_state_quant_param.first, activation_state_quant_param.second);
  activation_state_tensor.resize(activation_state_shape);
  Tensor output_tensor =
    makeOutputTensor(DataType::S8, output_quant_param.first, output_quant_param.second);

  Tensor scratchpad_activation_state(DataType::S16, Shape({}), {}, "");
  Tensor scratchpad_1(DataType::S32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::S32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_4(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_5(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_6(DataType::FLOAT32, Shape({}), {}, "");

  SVDFParams params{};
  params.activation = Activation::RELU;
  params.asymmetric_quantize_inputs = false;
  params.svdf_rank = rank;

  SVDF kernel(&input_tensor, &weight_feature_tensor, &weight_time_tensor, &bias_tensor,
              &activation_state_tensor, &output_tensor, &scratchpad_activation_state, &scratchpad_1,
              &scratchpad_2, &scratchpad_3, &scratchpad_4, &scratchpad_5, &scratchpad_6, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad_activation_state);
  _memory_manager->allocate_memory(scratchpad_1);
  _memory_manager->allocate_memory(scratchpad_2);
  _memory_manager->allocate_memory(scratchpad_3);
  _memory_manager->allocate_memory(scratchpad_4);
  _memory_manager->allocate_memory(scratchpad_5);
  _memory_manager->allocate_memory(scratchpad_6);
  kernel.execute();

  std::vector<int8_t> ref_output_data{-9, 24, 31, 1, -10, 10, -3, 0};

  std::vector<int32_t> ref_output_shape{batches, units};
  EXPECT_THAT(extractTensorData<int8_t>(output_tensor), ref_output_data);
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(SVDFTest, FloatTest)
{
  const int32_t batches = 2;
  const int32_t input_size = 3;
  const int32_t units = 4;
  const int32_t memory_size = 10;
  const int32_t rank = 1;
  const int32_t num_filters = units * rank;

  Shape input_shape{batches, input_size};
  Shape weight_feature_shape{num_filters, input_size};
  Shape weight_time_shape{num_filters, memory_size};
  Shape activation_state_shape{batches, memory_size * num_filters};

  std::vector<float> input_data{0.12609188, -0.46347019, -0.89598465,
                                0.35867718, 0.36897406,  0.73463392};

  std::vector<float> weight_feature_data{-0.31930989, -0.36118156, 0.0079667,   0.37613347,
                                         0.22197971,  0.12416199,  0.27901134,  0.27557442,
                                         0.3905206,   -0.36137494, -0.06634006, -0.10640851};

  std::vector<float> weight_time_data{
    -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
    0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

    0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
    -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

    -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
    0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

    -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
    -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor weight_feature_tensor = makeInputTensor<DataType::FLOAT32>(
    weight_feature_shape, weight_feature_data, _memory_manager.get());
  Tensor weight_time_tensor =
    makeInputTensor<DataType::FLOAT32>(weight_time_shape, weight_time_data, _memory_manager.get());
  Tensor activation_state_tensor = makeOutputTensor(DataType::FLOAT32);
  activation_state_tensor.resize(activation_state_shape);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_activation_state(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_4(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_5(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_6(DataType::FLOAT32, Shape({}), {}, "");

  SVDFParams params{};
  params.activation = Activation::NONE;
  params.asymmetric_quantize_inputs = false;
  params.svdf_rank = rank;

  SVDF kernel(&input_tensor, &weight_feature_tensor, &weight_time_tensor, nullptr,
              &activation_state_tensor, &output_tensor, &scratchpad_activation_state, &scratchpad_1,
              &scratchpad_2, &scratchpad_3, &scratchpad_4, &scratchpad_5, &scratchpad_6, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad_activation_state);
  _memory_manager->allocate_memory(scratchpad_1);
  _memory_manager->allocate_memory(scratchpad_2);
  _memory_manager->allocate_memory(scratchpad_3);
  _memory_manager->allocate_memory(scratchpad_4);
  _memory_manager->allocate_memory(scratchpad_5);
  _memory_manager->allocate_memory(scratchpad_6);
  kernel.execute();

  std::vector<float> ref_output_data{0.014899,    -0.0517661, -0.143725, -0.00271883,
                                     -0.03004015, 0.09565311, 0.1587342, 0.00784263};

  std::vector<float> ref_output_shape{batches, units};
  const float tolerance = 1e-5;
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data, tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(SVDFTest, Unsupported_Type_Configure_NEG)
{
  const int32_t batches = 2;
  const int32_t input_size = 3;
  const int32_t units = 4;
  const int32_t memory_size = 10;
  const int32_t rank = 1;
  const int32_t num_filters = units * rank;

  Shape input_shape{batches, input_size};
  Shape weight_feature_shape{num_filters, input_size};
  Shape weight_time_shape{num_filters, memory_size};
  Shape activation_state_shape{batches, memory_size * num_filters};

  std::vector<int32_t> input_data{0, 1, 3, 4, 4, -2};

  std::vector<float> weight_feature_data{-0.31930989, -0.36118156, 0.0079667,   0.37613347,
                                         0.22197971,  0.12416199,  0.27901134,  0.27557442,
                                         0.3905206,   -0.36137494, -0.06634006, -0.10640851};

  std::vector<float> weight_time_data{
    -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
    0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

    0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
    -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

    -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
    0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

    -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
    -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor weight_feature_tensor = makeInputTensor<DataType::FLOAT32>(
    weight_feature_shape, weight_feature_data, _memory_manager.get());
  Tensor weight_time_tensor =
    makeInputTensor<DataType::FLOAT32>(weight_time_shape, weight_time_data, _memory_manager.get());
  Tensor activation_state_tensor = makeOutputTensor(DataType::FLOAT32);
  activation_state_tensor.resize(activation_state_shape);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_activation_state(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_4(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_5(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_6(DataType::FLOAT32, Shape({}), {}, "");

  SVDFParams params{};
  params.activation = Activation::NONE;
  params.asymmetric_quantize_inputs = false;
  params.svdf_rank = rank;

  SVDF kernel(&input_tensor, &weight_feature_tensor, &weight_time_tensor, nullptr,
              &activation_state_tensor, &output_tensor, &scratchpad_activation_state, &scratchpad_1,
              &scratchpad_2, &scratchpad_3, &scratchpad_4, &scratchpad_5, &scratchpad_6, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(SVDFTest, Invalid_Input_Shape_NEG)
{
  const int32_t batches = 2;
  const int32_t right_input_size = 3;
  const int32_t wrong_input_size = 4;
  const int32_t units = 4;
  const int32_t memory_size = 10;
  const int32_t rank = 1;
  const int32_t num_filters = units * rank;

  Shape input_shape{batches, wrong_input_size};
  Shape weight_feature_shape{num_filters, right_input_size};
  Shape weight_time_shape{num_filters, memory_size};
  Shape activation_state_shape{batches, memory_size * num_filters};

  std::vector<float> input_data{0, 1, 3, 2, 4, 4, -2, 1};

  std::vector<float> weight_feature_data{-0.31930989, -0.36118156, 0.0079667,   0.37613347,
                                         0.22197971,  0.12416199,  0.27901134,  0.27557442,
                                         0.3905206,   -0.36137494, -0.06634006, -0.10640851};

  std::vector<float> weight_time_data{
    -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
    0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

    0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
    -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

    -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
    0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

    -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
    -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor weight_feature_tensor = makeInputTensor<DataType::FLOAT32>(
    weight_feature_shape, weight_feature_data, _memory_manager.get());
  Tensor weight_time_tensor =
    makeInputTensor<DataType::FLOAT32>(weight_time_shape, weight_time_data, _memory_manager.get());
  Tensor activation_state_tensor = makeOutputTensor(DataType::FLOAT32);
  activation_state_tensor.resize(activation_state_shape);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_activation_state(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_4(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_5(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_6(DataType::FLOAT32, Shape({}), {}, "");

  SVDFParams params{};
  params.activation = Activation::NONE;
  params.asymmetric_quantize_inputs = false;
  params.svdf_rank = rank;

  SVDF kernel(&input_tensor, &weight_feature_tensor, &weight_time_tensor, nullptr,
              &activation_state_tensor, &output_tensor, &scratchpad_activation_state, &scratchpad_1,
              &scratchpad_2, &scratchpad_3, &scratchpad_4, &scratchpad_5, &scratchpad_6, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
