/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/MirrorPad.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/pad.h>

namespace luci_interpreter
{
namespace kernels
{

MirrorPad::MirrorPad(const Tensor *input, const Tensor *paddings, Tensor *output,
                     const MirrorPadParams &params)
  : KernelWithParams<MirrorPadParams>({input, paddings}, {output}, params)
{
}

void MirrorPad::configure()
{
  const Shape &input_shape = input()->shape();
  const int num_dims = input_shape.num_dims();

  if (num_dims > 4)
    throw std::runtime_error("Unsupported number of dimensions.");

  assert(output()->element_type() == input()->element_type());
  assert(paddings()->element_type() == DataType::S32);
  // Paddings shape should be [N, 2].
  assert(paddings()->shape().num_dims() == 2);
  assert(paddings()->shape().dim(0) == num_dims);
  assert(paddings()->shape().dim(1) == 2);

  Shape output_shape(num_dims);
  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = 0; i < num_dims; ++i)
  {
    const int32_t padding_before = paddings_data[i * 2];
    const int32_t padding_after = paddings_data[i * 2 + 1];
    assert(padding_before >= 0 && padding_after >= 0);
    output_shape.dim(i) = input_shape.dim(i) + padding_before + padding_after;
  }

  output()->resize(output_shape);
}

template <typename T, typename P>
inline void MirrorPadImpl(const tflite::PadParams &op_params,
                          const tflite::RuntimeShape &input_shape, const T *input_data,
                          const P *pad_value_ptr, const tflite::RuntimeShape &output_shape,
                          T *output_data);

void MirrorPad::execute() const
{
  const int num_dims = input()->shape().num_dims();

  tflite::PadParams params{};
  params.left_padding_count = num_dims;
  params.right_padding_count = num_dims;

  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = num_dims - 1; i >= 0; --i)
  {
    params.left_padding[i] = paddings_data[i * 2];
    params.right_padding[i] = paddings_data[i * 2 + 1];
  }

  switch (input()->element_type())
  {
    case DataType::FLOAT32: {
      const float pad_value = 0;

      MirrorPadImpl(params, getTensorShape(input()), getTensorData<float>(input()), &pad_value,
                    getTensorShape(output()), getTensorData<float>(output()));
      break;
    }
    case DataType::U8: {
      assert(output()->zero_point() >= std::numeric_limits<uint8_t>::min());
      assert(output()->zero_point() <= std::numeric_limits<uint8_t>::max());
      const auto pad_value = static_cast<uint8_t>(output()->zero_point());
      MirrorPadImpl(params, getTensorShape(input()), getTensorData<uint8_t>(input()), &pad_value,
                    getTensorShape(output()), getTensorData<uint8_t>(output()));
      break;
    }
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

template <typename T, typename P>
inline void MirrorPadImpl(const tflite::PadParams &op_params,
                          const tflite::RuntimeShape &input_shape, const T *input_data,
                          const P *pad_value_ptr, const tflite::RuntimeShape &output_shape,
                          T *output_data)
{
  constexpr int32_t PAD_MAX_DIMENSION_COUNT = 5;
  const tflite::RuntimeShape ext_input_shape =
    tflite::RuntimeShape::ExtendedShape(PAD_MAX_DIMENSION_COUNT, input_shape);
  const tflite::RuntimeShape ext_output_shape =
    tflite::RuntimeShape::ExtendedShape(PAD_MAX_DIMENSION_COUNT, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, PAD_MAX_DIMENSION_COUNT);
  TFLITE_DCHECK_LE(op_params.right_padding_count, PAD_MAX_DIMENSION_COUNT);

  // Runtime calls are currently fixed at 5 dimensions. Copy inputs so we can
  // pad them to 5 dims (yes, we are "padding the padding").
  int32_t left_padding_copy[PAD_MAX_DIMENSION_COUNT];
  for (int32_t i = 0; i < PAD_MAX_DIMENSION_COUNT; i++)
  {
    left_padding_copy[i] = 0;
  }
  for (int32_t i = 0; i < op_params.left_padding_count; ++i)
  {
    left_padding_copy[i + PAD_MAX_DIMENSION_COUNT - op_params.left_padding_count] =
      op_params.left_padding[i];
  }
  int32_t right_padding_copy[PAD_MAX_DIMENSION_COUNT];
  for (int32_t i = 0; i < PAD_MAX_DIMENSION_COUNT; i++)
  {
    right_padding_copy[i] = 0;
  }
  for (int32_t i = 0; i < op_params.right_padding_count; ++i)
  {
    right_padding_copy[i + PAD_MAX_DIMENSION_COUNT - op_params.right_padding_count] =
      op_params.right_padding[i];
  }

  auto const input_batch = ext_input_shape.Dims(0);
  auto const input_plane = ext_input_shape.Dims(1);
  auto const input_height = ext_input_shape.Dims(2);
  auto const input_width = ext_input_shape.Dims(3);
  auto const input_depth = ext_input_shape.Dims(4);

  auto const input_h_offset = input_depth * input_width;
  auto const input_p_offset = input_h_offset * input_height;
  auto const input_b_offset = input_p_offset * input_plane;

  auto const output_batch = ext_output_shape.Dims(0);
  auto const output_plane = ext_output_shape.Dims(1);
  auto const output_height = ext_output_shape.Dims(2);
  auto const output_width = ext_output_shape.Dims(3);
  auto const output_depth = ext_output_shape.Dims(4);

  auto const left_b_padding = left_padding_copy[0];
  auto const left_p_padding = left_padding_copy[1];
  auto const left_h_padding = left_padding_copy[2];
  auto const left_w_padding = left_padding_copy[3];
  auto const left_d_padding = left_padding_copy[4];

  auto const right_b_padding = right_padding_copy[0];
  auto const right_p_padding = right_padding_copy[1];
  auto const right_h_padding = right_padding_copy[2];
  auto const right_w_padding = right_padding_copy[3];
  auto const right_d_padding = right_padding_copy[4];

  const auto positive_mod = [](auto a, auto b) { return (a % b + b) % b; };

  const T pad_value = *pad_value_ptr;

  const T *in_ptr = input_data;
  T *out_ptr = output_data;
  for (int32_t out_b = 0; out_b < output_batch; ++out_b)
  {
    for (int32_t out_p = 0; out_p < output_plane; ++out_p)
    {
      for (int32_t out_h = 0; out_h < output_height; ++out_h)
      {
        for (int32_t out_w = 0; out_w < output_width; ++out_w)
        {
          for (int32_t out_d = 0; out_d < output_depth; ++out_d)
          {
            if (out_b < left_b_padding || out_b >= output_batch - right_b_padding ||
                out_p < left_p_padding || out_p >= output_plane - right_p_padding ||
                out_h < left_h_padding || out_h >= output_height - right_h_padding ||
                out_w < left_w_padding || out_w >= output_width - right_w_padding ||
                out_d < left_d_padding || out_d >= output_depth - right_d_padding)
            {
              *out_ptr++ = *(input_data + positive_mod(out_d - left_d_padding, input_depth) +
                             positive_mod(out_w - left_w_padding, input_width) * input_depth +
                             positive_mod(out_h - left_h_padding, input_height) * input_h_offset +
                             positive_mod(out_p - left_p_padding, input_plane) * input_p_offset +
                             positive_mod(out_b - left_b_padding, input_batch) * input_b_offset);
            }
            else
            {
              *out_ptr++ = *in_ptr++;
            }
          }
        }
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
