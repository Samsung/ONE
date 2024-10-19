/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_CONV2D_H
#define LUCI_INTERPRETER_PAL_CONV2D_H

#include <tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h>
#include <tensorflow/lite/kernels/internal/reference/integer_ops/conv.h>
#include "HuffmanDecoder.h"

namespace luci_interpreter_pal
{
static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const float *input_data, const tflite::RuntimeShape &filter_shape,
                        const float *filter_data, const tflite::RuntimeShape &bias_shape,
                        const float *bias_data, const tflite::RuntimeShape &output_shape,
                        float *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        float *scratchpad_data)
{
  (void)scratchpad_shape;

  const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t input_depth = tflite::MatchingDim(input_shape, 3, filter_shape, 3);
  const int32_t output_height = output_shape.Dims(1);
  const int32_t output_width = output_shape.Dims(2);
  const int32_t filter_height = filter_shape.Dims(1);
  const int32_t filter_width = filter_shape.Dims(2);

  int64_t im2col_flat_size = 1;
  im2col_flat_size *= batches;
  im2col_flat_size *= output_height;
  im2col_flat_size *= output_width;
  im2col_flat_size *= input_depth;
  im2col_flat_size *= filter_height;
  im2col_flat_size *= filter_width;

  // This condition checks if integer overflow will occur inside the optimized kernel.
  // https://github.com/tensorflow/tensorflow/blob/v2.12.1/tensorflow/lite/kernels/internal/optimized/im2col_utils.h#L81
  // If overflow is expected, we fall back to the reference kernel.
  // NOTE This is just a rough check.
  bool opt_kernel_overflow = im2col_flat_size > std::numeric_limits<int32_t>::max();

  if (scratchpad_data and not opt_kernel_overflow)
  {
    tflite::RuntimeShape im2col_shape{batches, output_height, output_width,
                                      input_depth * filter_height * filter_width};

    tflite::optimized_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, output_shape, output_data, im2col_shape,
                                scratchpad_data);
  }
  else
    tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, output_shape, output_data,
                                tflite::RuntimeShape(), nullptr);
}

static inline void Conv(const tflite::ConvParams &params, const tflite::RuntimeShape &input_shape,
                        const uint8 *input_data, const tflite::RuntimeShape &filter_shape,
                        const uint8 *filter_data, const tflite::RuntimeShape &bias_shape,
                        const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                        uint8 *output_data, const tflite::RuntimeShape &scratchpad_shape,
                        uint8 *scratchpad_data)
{
  // TODO This should only be done once (although it takes only a few microseconds).
  //  Also, the user should be able to adjust the number of threads.
  auto gemmlowp_context = std::make_unique<gemmlowp::GemmContext>();
  gemmlowp_context->set_max_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

  tflite::reference_ops::Conv(params, input_shape, input_data, filter_shape, filter_data,
                              bias_shape, bias_data, output_shape, output_data, scratchpad_shape,
                              scratchpad_data, gemmlowp_context.get());
}

template <typename T>
void ConvPerChannelHuffman(const tflite::ConvParams &params, const int32_t *mult,
                           const int32_t *shifts, const tflite::RuntimeShape &input_shape,
                           const T *input_data, const tflite::RuntimeShape &filter_shape,
                           const T *filter_data, const tflite::RuntimeShape &bias_shape,
                           const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                           T *output_data, const tflite::RuntimeShape &scratchpad_shape,
                           T *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  // Get parameters.
  const int32_t input_offset = params.input_offset; // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  const int32_t filter_offset = params.weights_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data)
  {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  huffman::HuffmanDecoder<uint8_t> decoder;
  decoder.init_decoder(reinterpret_cast<const uint8_t *>(filter_data));
  decoder.reset_decode_idx();
  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
  {
    auto group = out_channel / filters_per_group;

    // extract compressed filter
    decoder.decode_n(reinterpret_cast<uint8_t *>(&scratchpad_data[0]), scratchpad_shape.FlatSize());

    for (int batch = 0; batch < batches; ++batch)
    {
      for (int out_y = 0; out_y < output_height; ++out_y)
      {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x)
        {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          int32_t acc = 0;

          for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel)
          {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y)
            {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x)
              {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;

                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);

                if (!is_point_inside_image)
                {
                  continue;
                }

                int32_t input_val = input_data[Offset(input_shape, batch, in_y, in_x,
                                                      in_channel + group * filter_input_depth)];
                int32_t filter_val =
                  scratchpad_data[(filter_y * filter_height + filter_x) * filter_width +
                                  in_channel];
                // Accumulate with 32 bits accumulator.
                // In the nudging process during model quantization, we force
                // real value of 0.0 be represented by a quantized value. This
                // guarantees that the input_offset is a int8_t, even though
                // it is represented using int32_t. int32_t += int8_t *
                // (int8_t - int8_t) so the highest value we can get from each
                // accumulation is [-127, 127] * ([-128, 127] -
                // [-128, 127]), which is [-32512, 32512]. log2(32512)
                // = 14.98, which means we can accumulate at least 2^16
                // multiplications without overflow. The accumulator is
                // applied to a filter so the accumulation logic will hold as
                // long as the filter size (filter_y * filter_x * in_channel)
                // does not exceed 2^16, which is the case in all the models
                // we have seen so far.
                // accumulator depth is smaller than 2^16.
                acc += (filter_val + filter_offset) * (input_val + input_offset);
              }
            }
          }

          if (bias_data)
          {
            acc += bias_data[out_channel];
          }
          acc = tflite::MultiplyByQuantizedMultiplier(acc, mult[out_channel], shifts[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<T>(acc);
        }
      }
    }
  }
}

static inline void ConvPerChannel(const tflite::ConvParams &params, const int32_t *mult,
                                  const int32_t *shifts, const tflite::RuntimeShape &input_shape,
                                  const int8 *input_data, const tflite::RuntimeShape &filter_shape,
                                  const int8 *filter_data, const tflite::RuntimeShape &bias_shape,
                                  const int32 *bias_data, const tflite::RuntimeShape &output_shape,
                                  int8 *output_data, const tflite::RuntimeShape &scratchpad_shape,
                                  int8 *scratchpad_data)
{
  (void)scratchpad_shape;
  (void)scratchpad_data;
  // TODO enable optimized version
  tflite::reference_integer_ops::ConvPerChannel(params, mult, shifts, input_shape, input_data,
                                                filter_shape, filter_data, bias_shape, bias_data,
                                                output_shape, output_data);
}

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *scratchpad,
                                         const luci_interpreter::DataType &input_data_type,
                                         const tflite::ConvParams &params,
                                         const tflite::RuntimeShape &input_shape,
                                         const tflite::RuntimeShape &filter_shape,
                                         const tflite::RuntimeShape &output_shape,
                                         bool is_compressed = false)
{
  const int32_t filter_height = filter_shape.Dims(1);
  const int32_t filter_width = filter_shape.Dims(2);

  // Allocate tensor for scratchpad, if needed.
  // The checks here should be aligned with the actual implementation.
  const bool need_dilated_scratchpad =
    params.dilation_height_factor != 1 || params.dilation_width_factor != 1;
  const bool need_non_dilated_scratchpad = params.stride_height != 1 || params.stride_width != 1 ||
                                           filter_height != 1 || filter_width != 1;
  auto _need_scratchpad = input_data_type != luci_interpreter::DataType::S16 &&
                          (need_dilated_scratchpad || need_non_dilated_scratchpad || is_compressed);

  if (_need_scratchpad)
  {
    const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int32_t input_depth = tflite::MatchingDim(input_shape, 3, filter_shape, 3);
    const int32_t output_height = output_shape.Dims(1);
    const int32_t output_width = output_shape.Dims(2);

    auto data_type_size = static_cast<int32_t>(luci_interpreter::getDataTypeSize(input_data_type));
    // im2col_shape
    // data_type_size is added because we use U8 for scratchpad buffer dtype
    luci_interpreter::Shape scratchpad_shape{batches, output_height, output_width,
                                             input_depth * filter_height * filter_width,
                                             data_type_size};
    scratchpad->resize(scratchpad_shape);
  }
  else
  {
    scratchpad->set_allocatable(false);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_CONV2D_H
