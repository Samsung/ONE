/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Builders.h"
#include "kernels/Utils.h"

#include "kernels/BinaryOpCommon.h"

#include "PALMul.h"

namespace luci_interpreter
{
namespace
{

#ifndef DIS_QUANT
template <typename T, typename Options = nullptr_t>
void evalTISOQuantizedS16Kernel(kernels::TISOKernel *kernel, kernels::TISOData *kernel_data,
                                const Options *options)
{
  const auto *input1 = kernel->input1();
  const auto *input2 = kernel->input2();
  const auto *output = kernel->output();

  const auto input1_scale = static_cast<double>(Tensor::scale(input1));
  const auto input2_scale = static_cast<double>(Tensor::scale(input2));
  const auto output_scale = static_cast<double>(Tensor::scale(output));

  constexpr int left_shift = 12;
  const double twice_max_input_scale = 2 * std::max(input1_scale, input2_scale);
  const double real_input1_multiplier = input1_scale / twice_max_input_scale;
  const double real_input2_multiplier = input2_scale / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * output_scale);

  int32_t input1_multiplier{}, input2_multiplier{}, output_multiplier{};
  int input1_shift{}, input2_shift{}, output_shift{};
  kernels::quantizeMultiplierSmallerThanOneExp(real_input1_multiplier, &input1_multiplier,
                                               &input1_shift);
  kernels::quantizeMultiplierSmallerThanOneExp(real_input2_multiplier, &input2_multiplier,
                                               &input2_shift);
  kernels::quantizeMultiplierSmallerThanOneExp(real_output_multiplier, &output_multiplier,
                                               &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  kernels::calculateActivationRangeQuantized(luci_actfunc(options->fused_activation_function()),
                                             output, &activation_min, &activation_max);

  auto fn = [output_multiplier, output_shift, activation_min, activation_max](int16_t input1_val,
                                                                              int16_t input2_val) {
    int32_t output = static_cast<int32_t>(input1_val) * static_cast<int32_t>(input2_val);
    output = tflite::MultiplyByQuantizedMultiplier(output, output_multiplier, output_shift);
    output = std::max(output, activation_min);
    output = std::min(output, activation_max);
    return static_cast<int16_t>(output);
  };

  kernels::BinaryOpBroadcastSlow(
    kernels::getTensorShape(input1), kernels::getTensorData<int16_t>(kernel_data->input1_data),
    kernels::getTensorShape(input2), kernels::getTensorData<int16_t>(kernel_data->input2_data),
    kernels::getTensorShape(output), kernels::getTensorData<int16_t>(kernel_data->output_data), fn);
}

template <typename T, typename Options = nullptr_t>
void evalTISOInplaceQuantizedS16Kernel(kernels::TISOKernel *kernel, const Options *options)
{
  uint8_t *inplace_data_ptr = nullptr;
  circle::Tensor *input_inplace_tensor = nullptr;

  kernels::TISOData kernel_data = kernel->readInplaceData(inplace_data_ptr, input_inplace_tensor);

  evalTISOQuantizedS16Kernel<T, Options>(kernel, &kernel_data, options);

  kernel->runtime_graph()->makeInplaceOperation(input_inplace_tensor, kernel->output());
  if (input_inplace_tensor == kernel->input1())
  {
    kernel->runtime_graph()->makeInplaceOperation(kernel->input2(), nullptr);
  }
  else
  {
    kernel->runtime_graph()->makeInplaceOperation(kernel->input1(), nullptr);
  }
}

#endif // DIS_QUANT

} // namespace

void configure_kernel_CircleMul(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input1()) ==
                         Tensor::element_type(kernel.input2()));
  LUCI_INTERPRETER_CHECK(Tensor::element_type(kernel.input1()) ==
                         Tensor::element_type(kernel.input2()));
#ifndef DIS_QUANT
  if (Tensor::element_type(kernel.input1()) == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(Tensor::zero_points(kernel.input1()).size() == 1 &&
                           Tensor::zero_points(kernel.input2()).size() == 1);
    LUCI_INTERPRETER_CHECK(Tensor::zero_point(kernel.input1()) == 0 &&
                           Tensor::zero_point(kernel.input2()) == 0 &&
                           Tensor::zero_point(kernel.output()) == 0);
  }
#endif // DIS_QUANT
}

void execute_kernel_CircleMul(const circle::Operator *cur_op, BaseRuntimeGraph *runtime_graph)
{
  kernels::TISOKernel kernel(cur_op, runtime_graph);

  const auto *options = cur_op->builtin_options_as_MulOptions();

  tflite::RuntimeShape input_shape1 =
    kernels::getTensorRuntimeShape(kernel.input1(), runtime_graph);
  tflite::RuntimeShape input_shape2 =
    kernels::getTensorRuntimeShape(kernel.input2(), runtime_graph);

  bool is_inplace = runtime_graph->is_inplace_op(cur_op);

  switch (Tensor::element_type(kernel.input1()))
  {
#ifndef DIS_FLOAT
    case DataType::FLOAT32:
    {
      auto tiso_func = luci_interpreter_pal::Mul<float>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastMul4DSlow<float>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<float>(tiso_func, broadcast_tiso_func, &kernel, options,
                                              std::move(input_shape1), std::move(input_shape2));
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<float>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                       options, std::move(input_shape1), std::move(input_shape2));
      }
    }
    break;
#endif // DIS_FLOAT
    case DataType::S64:
    {
      auto tiso_func = luci_interpreter_pal::Mul<int64_t>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastMul4DSlow<int64_t>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<int64_t>(tiso_func, broadcast_tiso_func, &kernel, options,
                                                std::move(input_shape1), std::move(input_shape2));
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<int64_t>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                         options, std::move(input_shape1), std::move(input_shape2));
      }
    }
    break;
    case DataType::S32:
    {
      auto tiso_func = luci_interpreter_pal::Mul<int32_t>;
      auto broadcast_tiso_func = luci_interpreter_pal::BroadcastMul4DSlow<int32_t>;
      if (is_inplace)
      {
        kernels::evalTISOInplaceKernel<int32_t>(tiso_func, broadcast_tiso_func, &kernel, options,
                                                std::move(input_shape1), std::move(input_shape2));
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOKernel<int32_t>(tiso_func, broadcast_tiso_func, &kernel, &kernel_data,
                                         options, std::move(input_shape1), std::move(input_shape2));
      }
    }
    break;
#ifndef DIS_QUANT
      // TODO: check quantize Mul
    case DataType::U8:
    {
      auto tiso_func = [](const tflite::ArithmeticParams &params,
                          const tflite::RuntimeShape &input1_shape, const uint8_t *input1_data,
                          const tflite::RuntimeShape &input2_shape, const uint8_t *input2_data,
                          const tflite::RuntimeShape &output_shape, uint8_t *output_data) {
        luci_interpreter_pal::Mul(params, input1_shape, input1_data, input2_shape, input2_data,
                                  output_shape, output_data);
      };
      auto broadcast_tiso_func =
        [](const tflite::ArithmeticParams &params, const tflite::RuntimeShape &input1_shape,
           const uint8_t *input1_data, const tflite::RuntimeShape &input2_shape,
           const uint8_t *input2_data, const tflite::RuntimeShape &output_shape,
           uint8_t *output_data) {
          luci_interpreter_pal::BroadcastMul4DSlow(params, input1_shape, input1_data, input2_shape,
                                                   input2_data, output_shape, output_data);
        };
      if (is_inplace)
      {
        kernels::evalTISOInplaceQuantizedKernel<uint8_t>(tiso_func, broadcast_tiso_func, &kernel,
                                                         options);
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        kernels::evalTISOQuantizedKernel<uint8_t>(tiso_func, broadcast_tiso_func, &kernel,
                                                  &kernel_data, options);
      }
    }
    break;
    case DataType::S16:
    {
      if (is_inplace)
      {
        evalTISOInplaceQuantizedS16Kernel<int16_t>(&kernel, options);
      }
      else
      {
        kernels::TISOData kernel_data = kernel.readData();
        evalTISOQuantizedS16Kernel<int16_t>(&kernel, &kernel_data, options);
      }
    }
    break;
#endif // DIS_QUANT
    default:
      assert(false && "Unsupported type.");
  }
}

} // namespace luci_interpreter
