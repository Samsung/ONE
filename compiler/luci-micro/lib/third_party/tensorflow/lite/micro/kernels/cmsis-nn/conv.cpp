/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "tensorflow/lite/micro/tools/make/downloads/cmsis//CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/cmsis-nn/scratch_buffer.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

const int kTensorNotAllocated = -1;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  // TODO(b/141139247): Allocate these dynamically when possible.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int num_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels));
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteConvParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* im2col, TfLiteTensor* hwcn_weights,
                           TfLiteTensor* output) {
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<uint8_t>(input), GetTensorShape(filter),
                      GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                      GetTensorData<int32_t>(bias), GetTensorShape(output),
                      GetTensorData<uint8_t>(output), GetTensorShape(im2col),
                      GetTensorData<uint8_t>(im2col), nullptr);
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, TfLiteConvParams* params,
    OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output, TfLiteTensor* im2col) {
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

#if defined(__ARM_FEATURE_DSP)
  RuntimeShape filter_shape = GetTensorShape(filter);
  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);
  RuntimeShape bias_shape = GetTensorShape(bias);

  // Set min and max value of the output.
  const int32 output_activation_min = std::numeric_limits<int8_t>::min();
  const int32 output_activation_max = std::numeric_limits<int8_t>::max();

  // Sanity check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (GetTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  int16_t* buf = nullptr;

  if (op_params.padding_values.width == 0 &&
      op_params.padding_values.height == 0 && (input_depth % 4 == 0) &&
      (output_depth % 2 == 0) && op_params.stride_width == 1 &&
      op_params.stride_height == 1 && filter_width == 1 && filter_height == 1) {
    const int32_t buf_size =
        arm_convolve_1x1_s8_fast_get_buffer_size(input_depth);
    if (get_cmsis_scratch_buffer(context, &buf, buf_size) != kTfLiteOk) {
      return kTfLiteError;
    }
    if (arm_convolve_1x1_s8_fast(
            GetTensorData<int8_t>(input), input_width, input_height,
            input_depth, batches, GetTensorData<int8_t>(filter), output_depth,
            op_params.padding_values.width, op_params.padding_values.height,
            op_params.stride_width, op_params.stride_height,
            GetTensorData<int32>(bias), GetTensorData<int8_t>(output),
            data->per_channel_output_shift, data->per_channel_output_multiplier,
            op_params.output_offset, op_params.input_offset,
            output_activation_min, output_activation_max, output_width,
            output_height, buf) != ARM_MATH_SUCCESS) {
      return kTfLiteError;
    }
  } else {
    const int32_t buf_size = arm_convolve_s8_get_buffer_size(
        input_depth, filter_width, filter_height);
    if (get_cmsis_scratch_buffer(context, &buf, buf_size) != kTfLiteOk) {
      return kTfLiteError;
    }
    if (arm_convolve_s8(
            GetTensorData<int8_t>(input), input_width, input_height,
            input_depth, batches, GetTensorData<int8_t>(filter), output_depth,
            filter_width, filter_height, op_params.padding_values.width,
            op_params.padding_values.height, op_params.stride_width,
            op_params.stride_height, GetTensorData<int32>(bias),
            GetTensorData<int8_t>(output), data->per_channel_output_shift,
            data->per_channel_output_multiplier, op_params.output_offset,
            op_params.input_offset, output_activation_min,
            output_activation_max, output_width, output_height,
            buf) != ARM_MATH_SUCCESS) {
      return kTfLiteError;
    }
  }
#else
#pragma message( \
    "CMSIS-NN optimization for conv not available for this target. Using reference kernel.")

  reference_integer_ops::ConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));

#endif
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* im2col,
                       TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<float>(input), GetTensorShape(filter),
                      GetTensorData<float>(filter), GetTensorShape(bias),
                      GetTensorData<float>(bias), GetTensorShape(output),
                      GetTensorData<float>(output), GetTensorShape(im2col),
                      GetTensorData<float>(im2col));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  OpData data;

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, &data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, &data, input, filter, bias,
                       nullptr, nullptr, output);
      break;
    case kTfLiteInt8:
      return EvalQuantizedPerChannel(context, node, params, &data, input,
                                     filter, bias, output, nullptr);
      break;
    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, &data, input, filter, bias,
                           nullptr, nullptr, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_CONV_2D() {
  static TfLiteRegistration r = {conv::Init, conv::Free, conv::Prepare,
                                 conv::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
