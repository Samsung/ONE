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

#include "kernels/SVDF.h"
#include "kernels/Utils.h"
#include "PALSVDF.h"

#include <tensorflow/lite/kernels/internal/quantization_util.h>

namespace luci_interpreter
{
namespace kernels
{

SVDF::SVDF(const Tensor *input, const Tensor *weight_feature, const Tensor *weight_time,
           const Tensor *bias, const Tensor *input_activation_state, Tensor *output,
           Tensor *scratchpad_activation_state, Tensor *scratchpad_1, Tensor *scratchpad_2,
           Tensor *scratchpad_3, Tensor *scratchpad_4, Tensor *scratchpad_5, Tensor *scratchpad_6,
           const SVDFParams &params)
  : KernelWithParams<SVDFParams>({input, weight_feature, weight_time, bias, input_activation_state},
                                 {output, scratchpad_activation_state, scratchpad_1, scratchpad_2,
                                  scratchpad_3, scratchpad_4, scratchpad_5, scratchpad_6},
                                 params)
{
  // Do nothing
}

void SVDF::configure()
{
  const Shape &input_shape = input()->shape();
  const Shape &weight_features_shape = weight_feature()->shape();
  const Shape &weight_time_shape = weight_time()->shape();

  // Validate Input Tensor:
  LUCI_INTERPRETER_CHECK(input()->element_type() == loco::DataType::FLOAT32 ||
                         input()->element_type() == loco::DataType::S8);
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() == 2);

  // Validate inputs and output types
  if (input()->element_type() == loco::DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(weight_feature()->element_type() == loco::DataType::S8);
    LUCI_INTERPRETER_CHECK(weight_time()->element_type() == loco::DataType::S16 ||
                           weight_time()->element_type() == loco::DataType::S8);
    if (bias())
      LUCI_INTERPRETER_CHECK(bias()->element_type() == loco::DataType::S32);

    LUCI_INTERPRETER_CHECK(input_activation_state()->element_type() == loco::DataType::S16 ||
                           input_activation_state()->element_type() == loco::DataType::S8);
    LUCI_INTERPRETER_CHECK(output()->element_type() == loco::DataType::S8);

    // Note: now tflite support only ReLU activation for integer SVDF
    LUCI_INTERPRETER_CHECK(params().activation == luci::FusedActFunc::RELU);
  }
  else if (weight_feature()->element_type() == loco::DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(weight_feature()->element_type() == loco::DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(weight_time()->element_type() == loco::DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(input_activation_state()->element_type() == loco::DataType::FLOAT32);
    if (bias())
      LUCI_INTERPRETER_CHECK(bias()->element_type() == loco::DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(output()->element_type() == loco::DataType::FLOAT32);
  }
  else if ((weight_feature()->element_type() == loco::DataType::U8 ||
            weight_feature()->element_type() == loco::DataType::S8) &&
           input()->element_type() == loco::DataType::FLOAT32)
  {
    // TODO:: support hybrid SVDF op
    throw std::runtime_error("Hybrid type is not currently supported");
  }
  else
  {
    throw std::runtime_error("luci-intp SVDF Unsupported type.");
  }

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params().svdf_rank;
  const int batch_size = input_shape.dim(0);
  const int num_filters = weight_features_shape.dim(0);
  LUCI_INTERPRETER_CHECK(rank != 0);
  LUCI_INTERPRETER_CHECK(num_filters % rank == 0);

  const int num_units = num_filters / rank;
  const int memory_size = weight_time_shape.dim(1);

  // Validate Weight_Feature Input Tensor:
  LUCI_INTERPRETER_CHECK(weight_features_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(weight_features_shape.dim(1) == input_shape.dim(1));

  // Validate Weight_Time Input Tensor:
  LUCI_INTERPRETER_CHECK(weight_time_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(weight_time_shape.dim(0) == num_filters);

  // Validate Bias
  if (bias())
    LUCI_INTERPRETER_CHECK(bias()->shape().dim(0) == num_units);

  // Validate Input Activation State
  LUCI_INTERPRETER_CHECK(input_activation_state()->shape().num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_activation_state()->shape().dim(0) == batch_size);
  LUCI_INTERPRETER_CHECK(input_activation_state()->shape().dim(1) == memory_size * num_filters);

  // Resize scratchpad_state to input_activation_state
  auto scratchpad_activation_state = getOutputTensors()[1];
  scratchpad_activation_state->resize({batch_size, memory_size * num_filters});

  // Resize output tensor
  output()->resize({batch_size, num_units});

  luci_interpreter_pal::SetupScratchpadTensor(
    input()->element_type(), weight_feature()->element_type(), getOutputTensors()[2],
    getOutputTensors()[3], getOutputTensors()[4], getOutputTensors()[5], getOutputTensors()[6],
    getOutputTensors()[7], input_shape, weight_time_shape, batch_size, num_filters, num_units);
}

void SVDF::execute() const
{
  switch (weight_feature()->element_type())
  {
    case loco::DataType::FLOAT32:
      evalFloat();
      break;
    case loco::DataType::S8:
    {
      if (input()->element_type() == loco::DataType::S8)
        evalInteger();
      else
        // TODO:: support hybrid SVDF op
        throw std::runtime_error("Hybrid type is not currently supported");
      break;
    }
    default:
      throw std::runtime_error("Unsupported type");
  }
}

void SVDF::evalInteger() const
{
  const auto effective_scale_1 = static_cast<double>(input()->scale() * weight_feature()->scale() /
                                                     input_activation_state()->scale());
  const auto effective_scale_2 = static_cast<double>(input_activation_state()->scale() *
                                                     weight_time()->scale() / output()->scale());

  int32_t effective_scale_1_a;
  int effective_scale_1_b;
  int32_t effective_scale_2_a;
  int effective_scale_2_b;

  tflite::QuantizeMultiplier(effective_scale_1, &effective_scale_1_a, &effective_scale_1_b);
  tflite::QuantizeMultiplier(effective_scale_2, &effective_scale_2_a, &effective_scale_2_b);

  TfLiteSVDFParams params_svdf{};
  params_svdf.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;
  params_svdf.rank = params().svdf_rank;
  params_svdf.activation = getTfLiteActivation(params().activation);

  auto scratchpad_activation_state = getOutputTensors()[1];
  // Note: it is expected that activation_state input variable tensor reset to zero,
  // also expected that this variable tensor doesn't have buffer
  auto scratchpad_data = getTensorData<int16_t>(scratchpad_activation_state);
  std::fill_n(scratchpad_data, scratchpad_activation_state->shape().num_elements(), 0);

  auto scratchpad = getOutputTensors()[2];
  auto output_temp = getOutputTensors()[3];

  int32_t input_zp = input()->zero_point();
  int32_t output_zp = output()->zero_point();
  luci_interpreter_pal::IntegerSVDF(
    params_svdf, getTensorShape(input()), getTensorData<int8_t>(input()),
    getTensorShape(weight_feature()), getTensorData<int8_t>(weight_feature()),
    getTensorShape(weight_time()), getTensorData<int16_t>(weight_time()), getTensorShape(bias()),
    getTensorData<int32_t>(bias()), scratchpad_data, getTensorShape(output()),
    getTensorData<int8_t>(output()), getTensorData<int32_t>(scratchpad),
    getTensorData<int32_t>(output_temp), effective_scale_1_a, effective_scale_1_b,
    effective_scale_2_a, effective_scale_2_b, input_zp, output_zp);
}

void SVDF::evalFloat() const
{
  TfLiteSVDFParams params_svdf{};
  params_svdf.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;
  params_svdf.rank = params().svdf_rank;
  params_svdf.activation = getTfLiteActivation(params().activation);

  auto scratchpad_activation_state = getOutputTensors()[1];
  // Note: it is expected that activation_state input variable tensor reset to zero,
  // also expected that this variable tensor doesn't have buffer
  auto scratchpad_data = getTensorData<float>(scratchpad_activation_state);
  std::fill_n(scratchpad_data, scratchpad_activation_state->shape().num_elements(), 0);

  auto scratchpad_1 = getOutputTensors()[2];

  luci_interpreter_pal::FloatSVDF(
    params_svdf, getTensorShape(input()), getTensorData<float>(input()),
    getTensorShape(weight_feature()), getTensorData<float>(weight_feature()),
    getTensorShape(weight_time()), getTensorData<float>(weight_time()), getTensorShape(bias()),
    getTensorData<float>(bias()), getTensorData<float>(scratchpad_1), scratchpad_data,
    getTensorShape(output()), getTensorData<float>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
