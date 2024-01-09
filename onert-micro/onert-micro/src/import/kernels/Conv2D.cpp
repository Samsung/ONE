/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "import/OMKernelConfigureBuilder.h"
#include "core/OMUtils.h"
#include "execute/OMUtils.h"
#include "core/OMShape.h"
#include "core/OMKernelData.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t weightTensorIdx = 1;
constexpr uint32_t biasTensorIdx = 2;

constexpr uint32_t outputTensorIdx = 0;

#ifndef DIS_QUANT
void calculateOpDataConv2D(const circle::Tensor *input, const circle::Tensor *weights,
                           const circle::Tensor *output, circle::ActivationFunctionType activation)
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;

  assert(input->quantization() != nullptr);                 // Fix caller
  assert(input->quantization()->scale()->size() == 1);      // Fix caller
  assert(input->quantization()->zero_point()->size() == 1); // Fix caller

  assert(weights->quantization() != nullptr);                 // Fix caller
  assert(weights->quantization()->scale()->size() == 1);      // Fix caller
  assert(weights->quantization()->zero_point()->size() == 1); // Fix caller

  assert(output->quantization() != nullptr);                 // Fix caller
  assert(output->quantization()->scale()->size() == 1);      // Fix caller
  assert(output->quantization()->zero_point()->size() == 1); // Fix caller

  const float input_scale = *input->quantization()->scale()->begin();
  const float weight_scale = *weights->quantization()->scale()->begin();
  const float output_scale = *output->quantization()->scale()->begin();

  const float output_zero_point = *output->quantization()->zero_point()->begin();

  real_multiplier =
    execute::getQuantizedConvolutionMultipler(input_scale, weight_scale, output_scale);
  execute::quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  execute::calculateActivationRangeQuantized(activation, output_zero_point, output_scale,
                                             output->type(), &output_activation_min,
                                             &output_activation_max);

  //  DataFullyConnected *op_params = new DataFullyConnected;
  //  op_params->output_multiplier = output_multiplier;
  //  op_params->output_shift = output_shift;

  // kernel.setKernelData(reinterpret_cast<uint8_t *>(op_params));
}
#endif

} // namespace

OMStatus onert_micro::import::configure_kernel_CircleConv2D(const OMConfigureArgs &config_args)
{
  OMRuntimeContext &runtime_context = config_args.runtime_context;
  uint16_t op_index = config_args.kernel_index;

  execute::OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);

  const circle::Tensor *input = runtime_kernel.inputs[inputTensorIdx];
  const circle::Tensor *weight = runtime_kernel.inputs[weightTensorIdx];
  const circle::Tensor *bias = runtime_kernel.inputs[biasTensorIdx];

  const circle::Tensor *output = runtime_kernel.outputs[outputTensorIdx];

  assert(input != nullptr);
  assert(weight != nullptr);
  // Bias can be nullptr
  assert(output != nullptr);

  OMStatus status = Ok;

  if ((input->type() == circle::TensorType_FLOAT32 &&
       weight->type() != circle::TensorType_FLOAT32) or
      (input->type() == circle::TensorType_INT8 && weight->type() != circle::TensorType_INT8) or
      (input->type() == circle::TensorType_INT16 && weight->type() != circle::TensorType_INT16))
  {
    return UnsupportedType;
  }

  core::OMShape input_shape(input);
  core::OMShape weight_shape(weight);
  core::OMShape bias_shape(bias);
  core::OMShape output_shape(output);

  status = utils::checkCondition(input_shape.rank() == 4);
  if (status != Ok)
    return status;

  status = utils::checkCondition(input_shape.rank() == output_shape.rank());
  if (status != Ok)
    return status;

  status = utils::checkCondition(input_shape.rank() == weight_shape.rank());
  if (status != Ok)
    return status;

  status =
    utils::checkCondition(bias == nullptr or weight_shape.dim(0) == bias_shape.num_elements());

  //  const auto option = runtime_kernel.first_operator->builtin_options_as_Conv2DOptions();
  //
  //  int32_t padding_h = 0;
  //  int32_t padding_w = 0;
  //
  //  const int input_width = input_shape.dim(2); //input->dims->data[2];
  //  const int input_height = input_shape.dim(1); //input->dims->data[1];
  //  const int weight_width = weight_shape.dim(2);// filter->dims->data[2];
  //  const int weight_height = weight_shape.dim(1); //filter->dims->data[1];
  //  execute::computePaddingHeightWidth(option->stride_h(), option->stride_w(),
  //  option->dilation_h_factor(),
  //                          option->dilation_w_factor(), input_height, input_width, weight_height,
  //                          weight_width,
  //                                     option->padding(), &padding_h, &padding_w);

  //  DataConv2D *data = new DataConv2D;
  //
  //  data->per_channel_output_multiplier.resize(0);
  //  data->per_channel_output_shift.resize(0);
  //  data->padding_w = padding_w;
  //  data->padding_h = padding_h;
  //
  //
  //  if (input->type() != circle::TensorType_FLOAT32)
  //  {
  //    // TODO: enable quant params
  //  }
  //
  //  runtime_storage.setKernelData(reinterpret_cast<uint8_t *>(data));

  return status;
}
