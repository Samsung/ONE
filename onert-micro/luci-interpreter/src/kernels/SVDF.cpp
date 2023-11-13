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

#include "Builders.h"
#include "kernels/Utils.h"

#include "PALSVDF.h"

namespace luci_interpreter
{

namespace
{
const int kSvdfInputTensor = 0;
const int kSvdfWeightsFeatureTensor = 1;
const int kSvdfWeightsTimeTensor = 2;
const int kSvdfBiasTensor = 3;
const int kSvdfInputActivationStateTensor =
  4; // This is a variable tensor, and will be modified by this op.
const int kSvdfOutputTensor = 0;
class SVDFKernel
{
public:
  SVDFKernel() = delete;

  explicit SVDFKernel(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
    : _runtime_graph(runtime_graph)
  {
    const auto input_index = cur_op->inputs()->operator[](kSvdfInputTensor);
    const auto weights_feature_index = cur_op->inputs()->operator[](kSvdfWeightsFeatureTensor);
    const auto weights_time_index = cur_op->inputs()->operator[](kSvdfWeightsTimeTensor);
    const auto bias_index = cur_op->inputs()->operator[](kSvdfBiasTensor);
    const auto activation_state_index =
      cur_op->inputs()->operator[](kSvdfInputActivationStateTensor);
    const auto output_index = cur_op->outputs()->operator[](kSvdfOutputTensor);

    assert(input_index != -1);
    assert(weights_feature_index != -1);
    assert(weights_time_index != -1);
    assert(activation_state_index != -1);
    assert(output_index != -1);

    _input_tensor = _runtime_graph->getCircleTensorByIndex(input_index);
    _weights_feature_tensor = _runtime_graph->getCircleTensorByIndex(weights_feature_index);
    _weights_time_tensor = _runtime_graph->getCircleTensorByIndex(weights_time_index);
    _bias_tensor = _runtime_graph->getCircleTensorByIndex(bias_index);
    _activation_state_tensor = _runtime_graph->getCircleTensorByIndex(activation_state_index);
    _output_tensor = _runtime_graph->getCircleTensorByIndex(output_index);

    assert(_input_tensor != nullptr);
    assert(_weights_feature_tensor != nullptr);
    assert(_weights_time_tensor != nullptr);
    assert(_activation_state_tensor != nullptr);
    assert(_output_tensor != nullptr);
  }

  const circle::Tensor *input() const { return _input_tensor; }
  const circle::Tensor *weights_feature() const { return _weights_feature_tensor; }
  const circle::Tensor *weights_time() const { return _weights_time_tensor; }
  const circle::Tensor *bias() const { return _bias_tensor; }
  const circle::Tensor *activation_state() const { return _activation_state_tensor; }
  const circle::Tensor *output() const { return _output_tensor; }

private:
  const circle::Tensor *_input_tensor;
  const circle::Tensor *_weights_feature_tensor;
  const circle::Tensor *_weights_time_tensor;
  const circle::Tensor *_bias_tensor;
  const circle::Tensor *_activation_state_tensor;
  const circle::Tensor *_output_tensor;

  BaseRuntimeGraph *_runtime_graph;
};
} // namespace

void configure_kernel_CircleSVDF(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}
  SVDFKernel kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_SVDFOptions();

  // Define input constants based on input tensor definition above:
  const int rank = options->rank();
  const int input_size = Tensor::dim(kernel.input(), 1);
  const int batch_size = Tensor::dim(kernel.input(), 0);
  const int num_filters = Tensor::dim(kernel.weights_feature(), 0);
  LUCI_INTERPRETER_CHECK(num_filters % rank == 0);

  const int num_units = num_filters / rank;
  const int memory_size = Tensor::dim(kernel.weights_time(), 1);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input()) == DataType::FLOAT32 or
                         Tensor::element_type(kernel.input()) == DataType::S8);
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.input()) == 2);

  // Validate Tensor Output:
  // [0] = float/int8_t, {2, batch_size, num_units}
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.output()) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.output(), 0) == batch_size);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.output(), 1) == num_units);

  // Validate Weights Feature Input Tensor
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.weights_feature()) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.weights_feature(), 1) == input_size);

  // Validate Weights Time Input Tensor:
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.weights_time()) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.weights_time(), 0) == num_filters);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.weights_time(), 1) == memory_size);

  // Validate Optional Bias Input Tensor:
  if (kernel.bias() != nullptr)
  {
    LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.bias(), 0) == num_units);
  }

  // Validate Activation State Input Tensor:
  LUCI_INTERPRETER_CHECK(Tensor::num_dims(kernel.activation_state()) == 2);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.activation_state(), 0) == batch_size);
  LUCI_INTERPRETER_CHECK(Tensor::dim(kernel.activation_state(), 1) == memory_size * num_filters);

  if (Tensor::element_type(kernel.input()) == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.weights_feature()) == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.weights_time()) == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.activation_state()) == DataType::FLOAT32);
    if (kernel.bias())
      LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.bias()) == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.output()) == DataType::FLOAT32);
  }
}

void execute_kernel_CircleSVDF(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  SVDFKernel kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_SVDFOptions();

  // Define input constants based on input tensor definition above:
  const int rank = options->rank();
  const int input_size = Tensor::dim(kernel.input(), 1);
  const int batch_size = Tensor::dim(kernel.input(), 0);
  const int num_filters = Tensor::dim(kernel.weights_feature(), 0);
  LUCI_INTERPRETER_CHECK(num_filters % rank == 0);

  const int num_units = num_filters / rank;
  const int memory_size = Tensor::dim(kernel.weights_time(), 1);

  const uint8_t *input_data = runtime_graph->getDataByTensor(kernel.input());
  const uint8_t *weights_feature_data =
    runtime_graph->getConstDataByTensor(kernel.weights_feature());
  const uint8_t *weights_time_data = runtime_graph->getConstDataByTensor(kernel.weights_time());
  const uint8_t *bias_data = runtime_graph->getConstDataByTensor(kernel.bias());
  uint8_t *output_data = runtime_graph->getDataByTensor(kernel.output());

  const auto type = Tensor::element_type(kernel.input());
  switch (type)
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
    {
      // Create and fill with 0 state tensor
      auto state_data = std::make_unique<float[]>(Tensor::num_elements(kernel.activation_state()));
      std::fill_n(state_data.get(), Tensor::num_elements(kernel.activation_state()), 0);

      auto scratch_data = std::make_unique<uint8_t[]>(batch_size * num_filters * sizeof(float));

      luci_interpreter_pal::SVDF(
        kernels::getTensorData<float>(input_data),
        kernels::getTensorData<float>(weights_feature_data),
        kernels::getTensorData<float>(weights_time_data), kernels::getTensorData<float>(bias_data),
        state_data.get(), kernels::getTensorData<float>(scratch_data.get()),
        kernels::getTensorData<float>(output_data), rank, input_size, batch_size, num_filters,
        num_units, memory_size, options->fused_activation_function());
    }
    break;
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
