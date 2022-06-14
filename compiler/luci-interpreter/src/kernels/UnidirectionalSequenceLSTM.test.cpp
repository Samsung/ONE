/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/UnidirectionalSequenceLSTM.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{
void quantizeFloat(const float *values, const int size, int8_t *quantized_values,
                   float *scaling_factor)
{
  auto minmax = std::minmax_element(values, values + size);
  auto min_value = *minmax.first;
  auto max_value = *minmax.second;

  const int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0)
  {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i)
  {
    const auto quantized_value = static_cast<int32_t>(std::round(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = static_cast<int8_t>(std::min(kScale, std::max(-kScale, quantized_value)));
  }
}

using namespace testing;

class UnidirectionalSequenceLSTMTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(UnidirectionalSequenceLSTMTest, NoCifgNoPeepholeNoProjectionNoClippingFloatTest)
{
  /*
   * use_cifg = false;
   * use_peephole = false;
   * use_projection_weights = false;
   * use_projection_bias = false;
   * is_layer_norm = false;
   */
  bool time_major = true;
  float cell_clip = 0.0;
  float proj_clip = 0.0;
  bool asymmetric_quantize_inputs = false;

  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;

  std::vector<float> input_to_input_weights_ = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                                0.04266912,  -0.15680569, -0.34856534, 0.43890524};
  std::vector<float> input_to_cell_weights_ = {-0.50013041, 0.1370284,  0.11810488, 0.2013163,
                                               -0.20583314, 0.44344562, 0.22077113, -0.29909778};
  std::vector<float> input_to_forget_weights_ = {0.09701663,  0.20334584, -0.50592935, -0.31343272,
                                                 -0.40032279, 0.44781327, 0.01387155,  -0.35593212};
  std::vector<float> input_to_output_weights_ = {-0.25065863, -0.28290087, 0.04613829, 0.40525138,
                                                 0.44272184,  0.03897077,  -0.1556896, 0.19487578};
  std::vector<float> input_gate_bias_ = {0., 0., 0., 0.};
  std::vector<float> cell_gate_bias_ = {0., 0., 0., 0.};
  std::vector<float> forget_gate_bias_ = {1., 1., 1., 1.};
  std::vector<float> output_gate_bias_ = {0., 0., 0., 0.};

  std::vector<float> recurrent_to_input_weights_ = {
    -0.0063535,  -0.2042388,  0.31454784,  -0.35746509, 0.28902304, 0.08183324,
    -0.16555229, 0.02286911,  -0.13566875, 0.03034258,  0.48091322, -0.12528998,
    0.24077177,  -0.51332325, -0.33502164, 0.10629296};

  std::vector<float> recurrent_to_cell_weights_ = {
    -0.3407414,  0.24443203,  -0.2078532,  0.26320225,  0.05695659, -0.00123841,
    -0.4744786,  -0.35869038, -0.06418842, -0.13502428, -0.501764,  0.22830659,
    -0.46367589, 0.26016325,  -0.03894562, -0.16368064};

  std::vector<float> recurrent_to_forget_weights_ = {
    -0.48684245, -0.06655136, 0.42224967,  0.2112639,   0.27654213, 0.20864892,
    -0.07646349, 0.45877004,  0.00141793,  -0.14609534, 0.36447752, 0.09196436,
    0.28053468,  0.01560611,  -0.20127171, -0.01140004};

  std::vector<float> recurrent_to_output_weights_ = {
    0.43385774,  -0.17194885, 0.2718237,  0.09215671,  0.24107647, -0.39835793,
    0.18212086,  0.01301402,  0.48572797, -0.50656658, 0.20047462, -0.20607421,
    -0.51818722, -0.15390486, 0.0468148,  0.39922136};

  std::vector<float> lstm_input_ = {{2., 3., 3., 4., 1., 1.}};
  std::vector<float> lstm_golden_output_ = {{-0.02973187, 0.1229473, 0.20885126, -0.15358765,
                                             -0.03716109, 0.12507336, 0.41193449, -0.20860538,
                                             -0.15053082, 0.09120187, 0.24278517, -0.12222792}};

  Tensor input = makeInputTensor<DataType::FLOAT32>({sequence_length, n_batch, n_input},
                                                    lstm_input_, _memory_manager.get());

  Tensor input_to_input_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_input}, input_to_input_weights_, _memory_manager.get());
  Tensor input_to_forget_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_input}, input_to_forget_weights_, _memory_manager.get());
  Tensor input_to_cell_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_input}, input_to_cell_weights_, _memory_manager.get());
  Tensor input_to_output_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_input}, input_to_output_weights_, _memory_manager.get());

  Tensor recurrent_to_input_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_output}, recurrent_to_input_weights_, _memory_manager.get());
  Tensor recurrent_to_forget_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_output}, recurrent_to_forget_weights_, _memory_manager.get());
  Tensor recurrent_to_cell_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_output}, recurrent_to_cell_weights_, _memory_manager.get());
  Tensor recurrent_to_output_weights = makeInputTensor<DataType::FLOAT32>(
    {n_cell, n_output}, recurrent_to_output_weights_, _memory_manager.get());

  Tensor *cell_to_input_weights = nullptr;
  Tensor *cell_to_forget_weights = nullptr;
  Tensor *cell_to_output_weights = nullptr;

  Tensor input_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, input_gate_bias_, _memory_manager.get());
  Tensor forget_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, forget_gate_bias_, _memory_manager.get());
  Tensor cell_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, cell_gate_bias_, _memory_manager.get());
  Tensor output_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, output_gate_bias_, _memory_manager.get());

  Tensor *projection_weights = nullptr;
  Tensor *projection_bias = nullptr;

  Tensor output_state = makeOutputTensor(loco::DataType::FLOAT32);
  output_state.resize({n_batch, n_output});
  Tensor cell_state = makeOutputTensor(loco::DataType::FLOAT32);
  cell_state.resize({n_batch, n_cell});

  _memory_manager->allocate_memory(output_state);
  _memory_manager->allocate_memory(cell_state);

  // Note: it is expected that output_state input variable tensor reset to zero
  auto output_state_data = output_state.data<float>();
  std::fill_n(output_state_data, output_state.shape().num_elements(), 0);

  // Note: it is expected that cell_state input variable tensor reset to zero
  auto cell_state_data = cell_state.data<float>();
  std::fill_n(cell_state_data, cell_state.shape().num_elements(), 0);

  Tensor *input_layer_norm_coefficients = nullptr;
  Tensor *forget_layer_norm_coefficients = nullptr;
  Tensor *cell_layer_norm_coefficients = nullptr;
  Tensor *output_layer_norm_coefficients = nullptr;

  Tensor output = makeOutputTensor(loco::DataType::FLOAT32);

  Tensor scratch_buffer(DataType::FLOAT32, Shape({}), {}, "");

  UnidirectionalSequenceLSTMParams params{};
  params.activation = luci::FusedActFunc::TANH;
  params.cell_clip = cell_clip;
  params.proj_clip = proj_clip;
  params.time_major = time_major;
  params.asymmetric_quantize_inputs = asymmetric_quantize_inputs;

  UnidirectionalSequenceLSTM kernel(
    &input, &input_to_input_weights, &input_to_forget_weights, &input_to_cell_weights,
    &input_to_output_weights, &recurrent_to_input_weights, &recurrent_to_forget_weights,
    &recurrent_to_cell_weights, &recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, &input_gate_bias, &forget_gate_bias,
    &cell_gate_bias, &output_gate_bias, projection_weights, projection_bias, &output_state,
    &cell_state, input_layer_norm_coefficients, forget_layer_norm_coefficients,
    cell_layer_norm_coefficients, output_layer_norm_coefficients, {&output, &scratch_buffer},
    params);
  kernel.configure();
  _memory_manager->allocate_memory(output);
  _memory_manager->allocate_memory(scratch_buffer);
  kernel.execute();

  std::vector<float> ref_output_shape{sequence_length, n_batch, n_output};
  const float tolerance = 1e-5;
  EXPECT_THAT(extractTensorShape(output), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear(lstm_golden_output_, tolerance));
}

TEST_F(UnidirectionalSequenceLSTMTest, NoCifgNoPeepholeNoProjectionNoClippingHybridTest)
{
  /*
   * use_cifg = false;
   * use_peephole = false;
   * use_projection_weights = false;
   * use_projection_bias = false;
   * is_layer_norm = false;
   */
  bool time_major = true;
  float cell_clip = 0.0;
  float proj_clip = 0.0;
  bool asymmetric_quantize_inputs = false;

  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;

  std::vector<float> input_to_input_weights_ = {-0.45018822, -0.02338299, -0.0870589,  -0.34550029,
                                                0.04266912,  -0.15680569, -0.34856534, 0.43890524};
  auto length = input_to_input_weights_.size();
  std::vector<int8_t> input_to_input_weights_q(length);
  float scaling_factor;
  quantizeFloat(input_to_input_weights_.data(), length, input_to_input_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> input_to_input_weights_q_p = {scaling_factor, 0};

  std::vector<float> input_to_cell_weights_ = {-0.50013041, 0.1370284,  0.11810488, 0.2013163,
                                               -0.20583314, 0.44344562, 0.22077113, -0.29909778};
  length = input_to_cell_weights_.size();
  std::vector<int8_t> input_to_cell_weights_q(length);
  quantizeFloat(input_to_cell_weights_.data(), length, input_to_cell_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> input_to_cell_weights_q_p = {scaling_factor, 0};

  std::vector<float> input_to_forget_weights_ = {0.09701663,  0.20334584, -0.50592935, -0.31343272,
                                                 -0.40032279, 0.44781327, 0.01387155,  -0.35593212};
  length = input_to_forget_weights_.size();
  std::vector<int8_t> input_to_forget_weights_q(length);
  quantizeFloat(input_to_forget_weights_.data(), length, input_to_forget_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> input_to_forget_weights_q_p = {scaling_factor, 0};

  std::vector<float> input_to_output_weights_ = {-0.25065863, -0.28290087, 0.04613829, 0.40525138,
                                                 0.44272184,  0.03897077,  -0.1556896, 0.19487578};
  length = input_to_output_weights_.size();
  std::vector<int8_t> input_to_output_weights_q(length);
  quantizeFloat(input_to_output_weights_.data(), length, input_to_output_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> input_to_output_weights_q_p = {scaling_factor, 0};

  std::vector<float> input_gate_bias_ = {0., 0., 0., 0.};
  std::vector<float> cell_gate_bias_ = {0., 0., 0., 0.};
  std::vector<float> forget_gate_bias_ = {1., 1., 1., 1.};
  std::vector<float> output_gate_bias_ = {0., 0., 0., 0.};

  std::vector<float> recurrent_to_input_weights_ = {
    -0.0063535,  -0.2042388,  0.31454784,  -0.35746509, 0.28902304, 0.08183324,
    -0.16555229, 0.02286911,  -0.13566875, 0.03034258,  0.48091322, -0.12528998,
    0.24077177,  -0.51332325, -0.33502164, 0.10629296};
  length = recurrent_to_input_weights_.size();
  std::vector<int8_t> recurrent_to_input_weights_q(length);
  quantizeFloat(recurrent_to_input_weights_.data(), length, recurrent_to_input_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> recurrent_to_input_weights_q_p = {scaling_factor, 0};

  std::vector<float> recurrent_to_cell_weights_ = {
    -0.3407414,  0.24443203,  -0.2078532,  0.26320225,  0.05695659, -0.00123841,
    -0.4744786,  -0.35869038, -0.06418842, -0.13502428, -0.501764,  0.22830659,
    -0.46367589, 0.26016325,  -0.03894562, -0.16368064};
  length = recurrent_to_cell_weights_.size();
  std::vector<int8_t> recurrent_to_cell_weights_q(length);
  quantizeFloat(recurrent_to_cell_weights_.data(), length, recurrent_to_cell_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> recurrent_to_cell_weights_q_p = {scaling_factor, 0};

  std::vector<float> recurrent_to_forget_weights_ = {
    -0.48684245, -0.06655136, 0.42224967,  0.2112639,   0.27654213, 0.20864892,
    -0.07646349, 0.45877004,  0.00141793,  -0.14609534, 0.36447752, 0.09196436,
    0.28053468,  0.01560611,  -0.20127171, -0.01140004};
  length = recurrent_to_forget_weights_.size();
  std::vector<int8_t> recurrent_to_forget_weights_q(length);
  quantizeFloat(recurrent_to_forget_weights_.data(), length, recurrent_to_forget_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> recurrent_to_forget_weights_q_p = {scaling_factor, 0};

  std::vector<float> recurrent_to_output_weights_ = {
    0.43385774,  -0.17194885, 0.2718237,  0.09215671,  0.24107647, -0.39835793,
    0.18212086,  0.01301402,  0.48572797, -0.50656658, 0.20047462, -0.20607421,
    -0.51818722, -0.15390486, 0.0468148,  0.39922136};
  length = recurrent_to_output_weights_.size();
  std::vector<int8_t> recurrent_to_output_weights_q(length);
  quantizeFloat(recurrent_to_output_weights_.data(), length, recurrent_to_output_weights_q.data(),
                &scaling_factor);
  std::pair<float, int32_t> recurrent_to_output_weights_q_p = {scaling_factor, 0};

  std::vector<float> lstm_input_ = {{2., 3., 3., 4., 1., 1.}};
  std::vector<float> lstm_golden_output_ = {{-0.02973187, 0.1229473, 0.20885126, -0.15358765,
                                             -0.03716109, 0.12507336, 0.41193449, -0.20860538,
                                             -0.15053082, 0.09120187, 0.24278517, -0.12222792}};

  Tensor input = makeInputTensor<DataType::FLOAT32>({sequence_length, n_batch, n_input},
                                                    lstm_input_, _memory_manager.get());

  Tensor input_to_input_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_input}, input_to_input_weights_q_p.first, input_to_input_weights_q_p.second,
    input_to_input_weights_q, _memory_manager.get());
  Tensor input_to_forget_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_input}, input_to_forget_weights_q_p.first, input_to_forget_weights_q_p.second,
    input_to_forget_weights_q, _memory_manager.get());
  Tensor input_to_cell_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_input}, input_to_cell_weights_q_p.first, input_to_cell_weights_q_p.second,
    input_to_cell_weights_q, _memory_manager.get());
  Tensor input_to_output_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_input}, input_to_output_weights_q_p.first, input_to_output_weights_q_p.second,
    input_to_output_weights_q, _memory_manager.get());

  Tensor recurrent_to_input_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_output}, recurrent_to_input_weights_q_p.first, recurrent_to_input_weights_q_p.second,
    recurrent_to_input_weights_q, _memory_manager.get());
  Tensor recurrent_to_forget_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_output}, recurrent_to_forget_weights_q_p.first,
    recurrent_to_forget_weights_q_p.second, recurrent_to_forget_weights_q, _memory_manager.get());
  Tensor recurrent_to_cell_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_output}, recurrent_to_cell_weights_q_p.first, recurrent_to_cell_weights_q_p.second,
    recurrent_to_cell_weights_q, _memory_manager.get());
  Tensor recurrent_to_output_weights = makeInputTensor<DataType::S8>(
    {n_cell, n_output}, recurrent_to_output_weights_q_p.first,
    recurrent_to_output_weights_q_p.second, recurrent_to_output_weights_q, _memory_manager.get());

  Tensor *cell_to_input_weights = nullptr;
  Tensor *cell_to_forget_weights = nullptr;
  Tensor *cell_to_output_weights = nullptr;

  Tensor input_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, input_gate_bias_, _memory_manager.get());
  Tensor forget_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, forget_gate_bias_, _memory_manager.get());
  Tensor cell_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, cell_gate_bias_, _memory_manager.get());
  Tensor output_gate_bias =
    makeInputTensor<DataType::FLOAT32>({n_cell}, output_gate_bias_, _memory_manager.get());

  Tensor *projection_weights = nullptr;
  Tensor *projection_bias = nullptr;

  Tensor output_state = makeOutputTensor(loco::DataType::FLOAT32);
  output_state.resize({n_batch, n_output});
  Tensor cell_state = makeOutputTensor(loco::DataType::FLOAT32);
  cell_state.resize({n_batch, n_cell});

  _memory_manager->allocate_memory(output_state);
  _memory_manager->allocate_memory(cell_state);

  // Note: it is expected that output_state input variable tensor reset to zero
  auto output_state_data = output_state.data<float>();
  std::fill_n(output_state_data, output_state.shape().num_elements(), 0);

  // Note: it is expected that cell_state input variable tensor reset to zero
  auto cell_state_data = cell_state.data<float>();
  std::fill_n(cell_state_data, cell_state.shape().num_elements(), 0);

  Tensor *input_layer_norm_coefficients = nullptr;
  Tensor *forget_layer_norm_coefficients = nullptr;
  Tensor *cell_layer_norm_coefficients = nullptr;
  Tensor *output_layer_norm_coefficients = nullptr;

  Tensor output = makeOutputTensor(loco::DataType::FLOAT32);

  std::vector<Tensor *> output_tensors;
  output_tensors.push_back(&output);

  Tensor scratch_buffer(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&scratch_buffer);
  Tensor input_quantized(DataType::S8, Shape({}), {}, "");
  output_tensors.push_back(&input_quantized);
  Tensor output_state_quantized(DataType::S8, Shape({}), {}, "");
  output_tensors.push_back(&output_state_quantized);
  Tensor cell_state_quantized(DataType::S8, Shape({}), {}, "");
  output_tensors.push_back(&cell_state_quantized);
  Tensor input_sf(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&input_sf);
  Tensor output_state_sf(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&output_state_sf);
  Tensor prod_scaling_factors(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&prod_scaling_factors);
  Tensor recovered_cell_weights(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&recovered_cell_weights);
  Tensor accum_scratch(DataType::S32, Shape({}), {}, "");
  output_tensors.push_back(&accum_scratch);
  Tensor input_zp(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&input_zp);
  Tensor output_state_zp(DataType::FLOAT32, Shape({}), {}, "");
  output_tensors.push_back(&output_state_zp);
  Tensor row_sums(DataType::S32, Shape({}), {}, "");
  output_tensors.push_back(&row_sums);

  UnidirectionalSequenceLSTMParams params{};
  params.activation = luci::FusedActFunc::TANH;
  params.cell_clip = cell_clip;
  params.proj_clip = proj_clip;
  params.time_major = time_major;
  params.asymmetric_quantize_inputs = asymmetric_quantize_inputs;

  UnidirectionalSequenceLSTM kernel(
    &input, &input_to_input_weights, &input_to_forget_weights, &input_to_cell_weights,
    &input_to_output_weights, &recurrent_to_input_weights, &recurrent_to_forget_weights,
    &recurrent_to_cell_weights, &recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, &input_gate_bias, &forget_gate_bias,
    &cell_gate_bias, &output_gate_bias, projection_weights, projection_bias, &output_state,
    &cell_state, input_layer_norm_coefficients, forget_layer_norm_coefficients,
    cell_layer_norm_coefficients, output_layer_norm_coefficients, std::move(output_tensors),
    params);
  kernel.configure();
  _memory_manager->allocate_memory(output);
  _memory_manager->allocate_memory(scratch_buffer);
  _memory_manager->allocate_memory(input_quantized);
  _memory_manager->allocate_memory(output_state_quantized);
  _memory_manager->allocate_memory(cell_state_quantized);
  _memory_manager->allocate_memory(input_sf);
  _memory_manager->allocate_memory(output_state_sf);
  _memory_manager->allocate_memory(prod_scaling_factors);
  _memory_manager->allocate_memory(recovered_cell_weights);
  _memory_manager->allocate_memory(accum_scratch);
  _memory_manager->allocate_memory(input_zp);
  _memory_manager->allocate_memory(output_state_zp);
  _memory_manager->allocate_memory(row_sums);
  kernel.execute();

  std::vector<float> ref_output_shape{sequence_length, n_batch, n_output};
  const float tolerance = 0.0157651;
  EXPECT_THAT(extractTensorShape(output), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(extractTensorData<float>(output), FloatArrayNear(lstm_golden_output_, tolerance));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
