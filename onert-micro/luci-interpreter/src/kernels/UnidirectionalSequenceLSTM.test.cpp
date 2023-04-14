/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#if 0
#include "kernels/UnidirectionalSequenceLSTM.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class UnidirectionalSequenceLSTMTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

// NOTE from NoCifgNoPeepholeNoProjectionNoClippingUnidirectionalLstmTest
TEST_F(UnidirectionalSequenceLSTMTest, FloatTest)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 2;
  const int32_t n_cell = 4;
  const int32_t n_output = 4;
  const int32_t sequence_length = 3;

  std::vector<float> input_to_input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                               0.04266912,  -0.15680569, -0.34856534, 0.43890524};

  std::vector<float> input_to_cell_weights = {-0.50013041, 0.1370284,  0.11810488, 0.2013163,
                                              -0.20583314, 0.44344562, 0.22077113, -0.29909778};

  std::vector<float> input_to_forget_weights = {0.09701663,  0.20334584, -0.50592935, -0.31343272,
                                                -0.40032279, 0.44781327, 0.01387155,  -0.35593212};

  std::vector<float> input_to_output_weights = {-0.25065863, -0.28290087, 0.04613829, 0.40525138,
                                                0.44272184,  0.03897077,  -0.1556896, 0.19487578};

  std::vector<float> input_gate_bias = {0., 0., 0., 0.};
  std::vector<float> forget_gate_bias = {1., 1., 1., 1.};
  std::vector<float> cell_gate_bias = {0., 0., 0., 0.};
  std::vector<float> output_gate_bias = {0., 0., 0., 0.};

  std::vector<float> recurrent_to_input_weights = {
    -0.0063535,  -0.2042388,  0.31454784,  -0.35746509, 0.28902304, 0.08183324,
    -0.16555229, 0.02286911,  -0.13566875, 0.03034258,  0.48091322, -0.12528998,
    0.24077177,  -0.51332325, -0.33502164, 0.10629296};

  std::vector<float> recurrent_to_forget_weights = {
    -0.48684245, -0.06655136, 0.42224967,  0.2112639,   0.27654213, 0.20864892,
    -0.07646349, 0.45877004,  0.00141793,  -0.14609534, 0.36447752, 0.09196436,
    0.28053468,  0.01560611,  -0.20127171, -0.01140004};

  std::vector<float> recurrent_to_cell_weights = {
    -0.3407414,  0.24443203,  -0.2078532,  0.26320225,  0.05695659, -0.00123841,
    -0.4744786,  -0.35869038, -0.06418842, -0.13502428, -0.501764,  0.22830659,
    -0.46367589, 0.26016325,  -0.03894562, -0.16368064};

  std::vector<float> recurrent_to_output_weights = {
    0.43385774,  -0.17194885, 0.2718237,  0.09215671,  0.24107647, -0.39835793,
    0.18212086,  0.01301402,  0.48572797, -0.50656658, 0.20047462, -0.20607421,
    -0.51818722, -0.15390486, 0.0468148,  0.39922136};

  Shape input_to_input_weights_shape{n_cell, n_input};
  Shape input_to_cell_weights_shape{n_cell, n_input};
  Shape input_to_forget_weights_shape{n_cell, n_input};
  Shape input_to_output_weights_shape{n_cell, n_input};

  Shape input_gate_bias_shape{n_cell};
  Shape forget_gate_bias_shape{n_cell};
  Shape cell_gate_bias_shape{n_cell};
  Shape output_gate_bias_shape{n_cell};

  Shape recurrent_to_input_weights_shape{n_cell, n_output};
  Shape recurrent_to_cell_weights_shape{n_cell, n_output};
  Shape recurrent_to_forget_weights_shape{n_cell, n_output};
  Shape recurrent_to_output_weights_shape{n_cell, n_output};

  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());
  Tensor input_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_cell_weights_shape, input_to_cell_weights, _memory_manager.get());
  Tensor input_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_forget_weights_shape, input_to_forget_weights, _memory_manager.get());
  Tensor input_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_output_weights_shape, input_to_output_weights, _memory_manager.get());

  Tensor input_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    input_gate_bias_shape, input_gate_bias, _memory_manager.get());
  Tensor forget_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    forget_gate_bias_shape, forget_gate_bias, _memory_manager.get());
  Tensor cell_gate_bias_tensor =
    makeInputTensor<DataType::FLOAT32>(cell_gate_bias_shape, cell_gate_bias, _memory_manager.get());
  Tensor output_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    output_gate_bias_shape, output_gate_bias, _memory_manager.get());

  Tensor recurrent_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_input_weights_shape, recurrent_to_input_weights, _memory_manager.get());
  Tensor recurrent_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_cell_weights_shape, recurrent_to_cell_weights, _memory_manager.get());
  Tensor recurrent_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_forget_weights_shape, recurrent_to_forget_weights, _memory_manager.get());
  Tensor recurrent_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_output_weights_shape, recurrent_to_output_weights, _memory_manager.get());

  std::vector<float> input_data{2., 3., 3., 4., 1., 1.};
  Shape input_shape{sequence_length, n_batch, n_input};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Shape output_state_shape{n_batch, n_output};
  Tensor output_state_tensor = makeOutputTensor(DataType::FLOAT32);
  output_state_tensor.resize(output_state_shape);

  Shape cell_state_shape{n_batch, n_cell};
  Tensor cell_state_tensor = makeOutputTensor(DataType::FLOAT32);
  cell_state_tensor.resize(cell_state_shape);

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 0.0;
  params.proj_clip = 0.0;
  params.time_major = true;
  params.asymmetric_quantize_inputs = false;

  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_forget_weights_tensor,
    &input_to_cell_weights_tensor, &input_to_output_weights_tensor,
    &recurrent_to_input_weights_tensor, &recurrent_to_forget_weights_tensor,
    &recurrent_to_cell_weights_tensor, &recurrent_to_output_weights_tensor, nullptr, nullptr,
    nullptr, &input_gate_bias_tensor, &forget_gate_bias_tensor, &cell_gate_bias_tensor,
    &output_gate_bias_tensor, nullptr, nullptr, &output_state_tensor, &cell_state_tensor, nullptr,
    nullptr, nullptr, nullptr, &output_tensor, &scratchpad_1, &scratchpad_2, &scratchpad_3, params);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(output_state_tensor);
  _memory_manager->allocate_memory(cell_state_tensor);
  _memory_manager->allocate_memory(scratchpad_1);
  _memory_manager->allocate_memory(scratchpad_2);
  _memory_manager->allocate_memory(scratchpad_3);
  kernel.execute();

  std::vector<float> ref_output_data{-0.02973187, 0.1229473,  0.20885126, -0.15358765,
                                     -0.03716109, 0.12507336, 0.41193449, -0.20860538,
                                     -0.15053082, 0.09120187, 0.24278517, -0.12222792};

  std::vector<float> ref_output_shape{sequence_length, n_batch, n_output};
  const float tolerance = 1e-5;
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data, tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(UnidirectionalSequenceLSTMTest, FloatTest_batch)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 2;
  const int32_t n_cell = 4;
  const int32_t n_output = 4;
  const int32_t sequence_length = 3;

  std::vector<float> input_to_input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                               0.04266912,  -0.15680569, -0.34856534, 0.43890524};

  std::vector<float> input_to_cell_weights = {-0.50013041, 0.1370284,  0.11810488, 0.2013163,
                                              -0.20583314, 0.44344562, 0.22077113, -0.29909778};

  std::vector<float> input_to_forget_weights = {0.09701663,  0.20334584, -0.50592935, -0.31343272,
                                                -0.40032279, 0.44781327, 0.01387155,  -0.35593212};

  std::vector<float> input_to_output_weights = {-0.25065863, -0.28290087, 0.04613829, 0.40525138,
                                                0.44272184,  0.03897077,  -0.1556896, 0.19487578};

  std::vector<float> input_gate_bias = {0., 0., 0., 0.};
  std::vector<float> forget_gate_bias = {1., 1., 1., 1.};
  std::vector<float> cell_gate_bias = {0., 0., 0., 0.};
  std::vector<float> output_gate_bias = {0., 0., 0., 0.};

  std::vector<float> recurrent_to_input_weights = {
    -0.0063535,  -0.2042388,  0.31454784,  -0.35746509, 0.28902304, 0.08183324,
    -0.16555229, 0.02286911,  -0.13566875, 0.03034258,  0.48091322, -0.12528998,
    0.24077177,  -0.51332325, -0.33502164, 0.10629296};

  std::vector<float> recurrent_to_forget_weights = {
    -0.48684245, -0.06655136, 0.42224967,  0.2112639,   0.27654213, 0.20864892,
    -0.07646349, 0.45877004,  0.00141793,  -0.14609534, 0.36447752, 0.09196436,
    0.28053468,  0.01560611,  -0.20127171, -0.01140004};

  std::vector<float> recurrent_to_cell_weights = {
    -0.3407414,  0.24443203,  -0.2078532,  0.26320225,  0.05695659, -0.00123841,
    -0.4744786,  -0.35869038, -0.06418842, -0.13502428, -0.501764,  0.22830659,
    -0.46367589, 0.26016325,  -0.03894562, -0.16368064};

  std::vector<float> recurrent_to_output_weights = {
    0.43385774,  -0.17194885, 0.2718237,  0.09215671,  0.24107647, -0.39835793,
    0.18212086,  0.01301402,  0.48572797, -0.50656658, 0.20047462, -0.20607421,
    -0.51818722, -0.15390486, 0.0468148,  0.39922136};

  Shape input_to_input_weights_shape{n_cell, n_input};
  Shape input_to_cell_weights_shape{n_cell, n_input};
  Shape input_to_forget_weights_shape{n_cell, n_input};
  Shape input_to_output_weights_shape{n_cell, n_input};

  Shape input_gate_bias_shape{n_cell};
  Shape forget_gate_bias_shape{n_cell};
  Shape cell_gate_bias_shape{n_cell};
  Shape output_gate_bias_shape{n_cell};

  Shape recurrent_to_input_weights_shape{n_cell, n_output};
  Shape recurrent_to_cell_weights_shape{n_cell, n_output};
  Shape recurrent_to_forget_weights_shape{n_cell, n_output};
  Shape recurrent_to_output_weights_shape{n_cell, n_output};

  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());
  Tensor input_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_cell_weights_shape, input_to_cell_weights, _memory_manager.get());
  Tensor input_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_forget_weights_shape, input_to_forget_weights, _memory_manager.get());
  Tensor input_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_output_weights_shape, input_to_output_weights, _memory_manager.get());

  Tensor input_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    input_gate_bias_shape, input_gate_bias, _memory_manager.get());
  Tensor forget_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    forget_gate_bias_shape, forget_gate_bias, _memory_manager.get());
  Tensor cell_gate_bias_tensor =
    makeInputTensor<DataType::FLOAT32>(cell_gate_bias_shape, cell_gate_bias, _memory_manager.get());
  Tensor output_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    output_gate_bias_shape, output_gate_bias, _memory_manager.get());

  Tensor recurrent_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_input_weights_shape, recurrent_to_input_weights, _memory_manager.get());
  Tensor recurrent_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_cell_weights_shape, recurrent_to_cell_weights, _memory_manager.get());
  Tensor recurrent_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_forget_weights_shape, recurrent_to_forget_weights, _memory_manager.get());
  Tensor recurrent_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_output_weights_shape, recurrent_to_output_weights, _memory_manager.get());

  std::vector<float> input_data{2., 3., 3., 4., 1., 1.};
  Shape input_shape{n_batch, sequence_length, n_input};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Shape output_state_shape{n_batch, n_output};
  Tensor output_state_tensor = makeOutputTensor(DataType::FLOAT32);
  output_state_tensor.resize(output_state_shape);

  Shape cell_state_shape{n_batch, n_cell};
  Tensor cell_state_tensor = makeOutputTensor(DataType::FLOAT32);
  cell_state_tensor.resize(cell_state_shape);

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 0.0;
  params.proj_clip = 0.0;
  params.time_major = false;
  params.asymmetric_quantize_inputs = false;

  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_forget_weights_tensor,
    &input_to_cell_weights_tensor, &input_to_output_weights_tensor,
    &recurrent_to_input_weights_tensor, &recurrent_to_forget_weights_tensor,
    &recurrent_to_cell_weights_tensor, &recurrent_to_output_weights_tensor, nullptr, nullptr,
    nullptr, &input_gate_bias_tensor, &forget_gate_bias_tensor, &cell_gate_bias_tensor,
    &output_gate_bias_tensor, nullptr, nullptr, &output_state_tensor, &cell_state_tensor, nullptr,
    nullptr, nullptr, nullptr, &output_tensor, &output_state_tensor, &cell_state_tensor,
    &scratchpad_1, params);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(output_state_tensor);
  _memory_manager->allocate_memory(cell_state_tensor);
  _memory_manager->allocate_memory(scratchpad_1);
  kernel.execute();

  std::vector<float> ref_output_data{-0.02973187, 0.1229473,  0.20885126, -0.15358765,
                                     -0.03716109, 0.12507336, 0.41193449, -0.20860538,
                                     -0.15053082, 0.09120187, 0.24278517, -0.12222792};

  std::vector<float> ref_output_shape{n_batch, sequence_length, n_output};
  const float tolerance = 1e-5;
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data, tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(UnidirectionalSequenceLSTMTest, FloatTest_simple)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 1;
  const int32_t n_cell = 1;
  const int32_t n_output = 1;
  const int32_t sequence_length = 1;

  std::vector<float> input_to_input_weights = {0.329067};
  std::vector<float> input_to_forget_weights = {0.308059};
  std::vector<float> input_to_cell_weights = {0.152916};
  std::vector<float> input_to_output_weights = {-0.476033};

  std::vector<float> input_gate_bias = {0.};
  std::vector<float> forget_gate_bias = {1.};
  std::vector<float> cell_gate_bias = {0.};
  std::vector<float> output_gate_bias = {0.};

  std::vector<float> recurrent_to_input_weights = {0.207806};
  std::vector<float> recurrent_to_forget_weights = {0.028718};
  std::vector<float> recurrent_to_cell_weights = {-0.182756};
  std::vector<float> recurrent_to_output_weights = {-0.960517};

  Shape input_to_input_weights_shape{n_cell, n_input};
  Shape input_to_cell_weights_shape{n_cell, n_input};
  Shape input_to_forget_weights_shape{n_cell, n_input};
  Shape input_to_output_weights_shape{n_cell, n_input};

  Shape input_gate_bias_shape{n_cell};
  Shape forget_gate_bias_shape{n_cell};
  Shape cell_gate_bias_shape{n_cell};
  Shape output_gate_bias_shape{n_cell};

  Shape recurrent_to_input_weights_shape{n_cell, n_output};
  Shape recurrent_to_cell_weights_shape{n_cell, n_output};
  Shape recurrent_to_forget_weights_shape{n_cell, n_output};
  Shape recurrent_to_output_weights_shape{n_cell, n_output};

  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());
  Tensor input_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_cell_weights_shape, input_to_cell_weights, _memory_manager.get());
  Tensor input_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_forget_weights_shape, input_to_forget_weights, _memory_manager.get());
  Tensor input_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_output_weights_shape, input_to_output_weights, _memory_manager.get());

  Tensor input_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    input_gate_bias_shape, input_gate_bias, _memory_manager.get());
  Tensor forget_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    forget_gate_bias_shape, forget_gate_bias, _memory_manager.get());
  Tensor cell_gate_bias_tensor =
    makeInputTensor<DataType::FLOAT32>(cell_gate_bias_shape, cell_gate_bias, _memory_manager.get());
  Tensor output_gate_bias_tensor = makeInputTensor<DataType::FLOAT32>(
    output_gate_bias_shape, output_gate_bias, _memory_manager.get());

  Tensor recurrent_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_input_weights_shape, recurrent_to_input_weights, _memory_manager.get());
  Tensor recurrent_to_cell_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_cell_weights_shape, recurrent_to_cell_weights, _memory_manager.get());
  Tensor recurrent_to_forget_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_forget_weights_shape, recurrent_to_forget_weights, _memory_manager.get());
  Tensor recurrent_to_output_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    recurrent_to_output_weights_shape, recurrent_to_output_weights, _memory_manager.get());

  std::vector<float> input_data{0.03653763};
  Shape input_shape{n_batch, sequence_length, n_input};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  Shape output_state_shape{n_batch, n_output};
  Tensor output_state_tensor = makeOutputTensor(DataType::FLOAT32);
  output_state_tensor.resize(output_state_shape);

  Shape cell_state_shape{n_batch, n_cell};
  Tensor cell_state_tensor = makeOutputTensor(DataType::FLOAT32);
  cell_state_tensor.resize(cell_state_shape);

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 10.0;
  params.proj_clip = 0.0;
  params.time_major = false;
  params.asymmetric_quantize_inputs = false;

  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_forget_weights_tensor,
    &input_to_cell_weights_tensor, &input_to_output_weights_tensor,
    &recurrent_to_input_weights_tensor, &recurrent_to_forget_weights_tensor,
    &recurrent_to_cell_weights_tensor, &recurrent_to_output_weights_tensor, nullptr, nullptr,
    nullptr, &input_gate_bias_tensor, &forget_gate_bias_tensor, &cell_gate_bias_tensor,
    &output_gate_bias_tensor, nullptr, nullptr, &output_state_tensor, &cell_state_tensor, nullptr,
    nullptr, nullptr, nullptr, &output_tensor, &output_state_tensor, &cell_state_tensor,
    &scratchpad_1, params);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(output_state_tensor);
  _memory_manager->allocate_memory(cell_state_tensor);
  _memory_manager->allocate_memory(scratchpad_1);
  kernel.execute();

  std::vector<float> ref_output_data{0.00139296};
  std::vector<float> ref_output_shape{n_batch, sequence_length, n_output};
  const float tolerance = 1e-5;
  auto aa = extractTensorData<float>(output_tensor);
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data, tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(UnidirectionalSequenceLSTMTest, Unsupported_Type_Configure_NEG)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 2;
  const int32_t n_cell = 4;
  const int32_t n_output = 4;
  const int32_t sequence_length = 3;

  std::vector<int8_t> input_data{2, 3, 3, 4, 1, 1}; // int8 is not support as of now
  Shape input_shape{sequence_length, n_batch, n_input};
  Tensor input_tensor =
    makeInputTensor<DataType::S8>(input_shape, input_data, _memory_manager.get());

  std::vector<float> input_to_input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                               0.04266912,  -0.15680569, -0.34856534, 0.43890524};
  Shape input_to_input_weights_shape{n_cell, n_input};
  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 0.0;
  params.proj_clip = 0.0;
  params.time_major = true;
  params.asymmetric_quantize_inputs = false;

  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    nullptr, nullptr, nullptr, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr, nullptr,
    nullptr, &output_tensor, &scratchpad_1, &scratchpad_2, &scratchpad_3, params);

  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(UnidirectionalSequenceLSTMTest, Invalid_Input_Shape_NEG)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 2;
  const int32_t n_cell = 4;
  const int32_t n_output = 4;
  const int32_t sequence_length = 3;

  std::vector<float> input_data{2., 3., 3., 4., 1., 1.};
  Shape input_shape{sequence_length, n_input}; // this is wrong
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  std::vector<float> input_to_input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                               0.04266912,  -0.15680569, -0.34856534, 0.43890524};
  Shape input_to_input_weights_shape{n_cell, n_input};
  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 0.0;
  params.proj_clip = 0.0;
  params.time_major = true;
  params.asymmetric_quantize_inputs = false;

  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    nullptr, nullptr, nullptr, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr, nullptr,
    nullptr, &output_tensor, &scratchpad_1, &scratchpad_2, &scratchpad_3, params);

  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(UnidirectionalSequenceLSTMTest, Invalid_Input_Shape_2_NEG)
{
  const int32_t n_batch = 1;
  const int32_t n_input = 2;
  const int32_t n_cell = 4;
  const int32_t n_output = 4;
  const int32_t sequence_length = 3;

  std::vector<float> input_data{2., 3., 3., 4., 1., 1.};
  Shape input_shape{sequence_length, n_batch, n_input};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());

  std::vector<float> input_to_input_weights = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                               0.04266912,  -0.15680569, -0.34856534, 0.43890524};
  Shape input_to_input_weights_shape{n_cell, n_input};
  Tensor input_to_input_weights_tensor = makeInputTensor<DataType::FLOAT32>(
    input_to_input_weights_shape, input_to_input_weights, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad_1(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_2(DataType::FLOAT32, Shape({}), {}, "");
  Tensor scratchpad_3(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = Activation::TANH;
  params.cell_clip = 0.0;
  params.proj_clip = 0.0;
  params.time_major = true;
  params.asymmetric_quantize_inputs = false;

  // NOTE provide wrong shaped inputs
  UnidirectionalSequenceLSTM kernel(
    &input_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    nullptr, nullptr, nullptr, &input_to_input_weights_tensor, &input_to_input_weights_tensor,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr,
    &input_to_input_weights_tensor, &input_to_input_weights_tensor, nullptr, nullptr, nullptr,
    nullptr, &output_tensor, &scratchpad_1, &scratchpad_2, &scratchpad_3, params);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
