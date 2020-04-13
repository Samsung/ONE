/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Pad.h"
#include "Assert.h"
#include "Logging.h"

#include "internal/Dims.h"

#include <vector>
#include <cstring> // For 'memset'

bool padPrepare(const Shape& input, const int32_t* paddingsData,  const Shape& paddingsShape,
                Shape* output)
{
  // Currently only 4D tensors are supported.
  uint32_t numInputDims = getNumberOfDimensions(input);
  ASSERT(numInputDims == 4);

  // paddings need to be provided as a 2-D int32 tensor.
  ASSERT(paddingsShape.type == OperandType::TENSOR_INT32);
  ASSERT(getNumberOfDimensions(paddingsShape) == 2);
  ASSERT(getSizeOfDimension(paddingsShape, 0) == numInputDims);
  ASSERT(getSizeOfDimension(paddingsShape, 1) == 2);

  std::vector<uint32_t> outDims(numInputDims);
  for (uint32_t i = 0; i < numInputDims; ++i)
  {
    int32_t beforePadding = *paddingsData++;
    int32_t afterPadding = *paddingsData++;
    // Pad value has to be greater than equal to 0.
    ASSERT(beforePadding >= 0 && afterPadding >= 0);
    outDims[i] = beforePadding + getSizeOfDimension(input, i) + afterPadding;
  }
  output->type = input.type;
  output->dimensions = outDims;
  output->offset = input.offset;
  output->scale = input.scale;

  return true;
}

namespace
{

// From optimized_ops.h in TensorFlow Lite
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  const int output_batch = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int output_depth = ArraySize(output_dims, 0);

  const int left_b_padding = left_paddings[3];
  const int left_h_padding = left_paddings[2];
  const int left_w_padding = left_paddings[1];
  const int left_d_padding = left_paddings[0];

  const int right_b_padding = right_paddings[3];
  const int right_h_padding = right_paddings[2];
  const int right_w_padding = right_paddings[1];
  const int right_d_padding = right_paddings[0];

  const int input_depth = ArraySize(input_dims, 0);

  if (left_b_padding != 0)
  {
    memset(output_data, 0, left_b_padding * output_height * output_width * output_depth *
           sizeof(T));
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding; ++out_b)
  {
    if (left_h_padding != 0)
    {
      memset(output_data + Offset(output_dims, 0, 0, 0, out_b), 0,
             left_h_padding * output_width * output_depth * sizeof(T));
    }
    for (int out_h = left_h_padding; out_h < output_height - right_h_padding; ++out_h)
    {
      if (left_w_padding != 0)
      {
        memset(output_data + Offset(output_dims, 0, 0, out_h, out_b), 0,
               left_w_padding * output_depth * sizeof(T));
      }
      for (int out_w = left_w_padding; out_w < output_width - right_w_padding; ++out_w)
      {
        if (left_d_padding != 0)
        {
          memset(output_data + Offset(output_dims, 0, out_w, out_h, out_b), 0,
                 left_d_padding * sizeof(T));
        }

        T* out = output_data +
                 Offset(output_dims, left_d_padding, out_w, out_h, out_b);
        const T* in =
            input_data + Offset(input_dims, 0, out_w - left_w_padding,
                                out_h - left_h_padding, out_b - left_b_padding);
        memcpy(out, in, input_depth * sizeof(T));

        if (right_d_padding != 0)
        {
          memset(
              output_data + Offset(output_dims, output_depth - right_d_padding,
                                   out_w, out_h, out_b),
              0, right_d_padding * sizeof(T));
        }
      }
      if (right_w_padding != 0)
      {
        memset(
            output_data + Offset(output_dims, 0, output_width - right_w_padding,
                                 out_h, out_b),
            0, right_w_padding * output_depth * sizeof(T));
      }
    }
    if (right_h_padding != 0)
    {
      memset(output_data + Offset(output_dims, 0, 0,
                                  output_height - right_h_padding, out_b),
             0, right_h_padding * output_width * output_depth * sizeof(T));
    }
  }
  if (right_b_padding != 0)
  {
    memset(output_data +
               Offset(output_dims, 0, 0, 0, output_batch - right_b_padding),
           0,
           right_b_padding * output_height * output_width * output_depth *
               sizeof(T));
  }
}

} // namespace

bool padGeneric(const uint8_t* inputData, const Shape& inputShape, const int32_t* paddings,
                uint8_t* outputData, const Shape& outputShape)
{
  int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(inputShape));

  std::vector<int> beforePadding;
  std::vector<int> afterPadding;
  // The lower level implementation expects the paddings in the reverse order.
  for (int32_t i = numInputDims - 1; i >= 0; --i)
  {
    beforePadding.push_back(paddings[i * 2]);
    afterPadding.push_back(paddings[i * 2 + 1]);
  }

  if (inputShape.type == OperandType::TENSOR_FLOAT32)
  {
    ::Pad(reinterpret_cast<const float*>(inputData),
          convertShapeToDims(inputShape),
          beforePadding, afterPadding,
          reinterpret_cast<float*>(outputData),
          convertShapeToDims(outputShape));
  }
  else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM)
  {
    ::Pad(reinterpret_cast<const uint8_t*>(inputData),
          convertShapeToDims(inputShape),
          beforePadding, afterPadding,
          reinterpret_cast<uint8_t*>(outputData),
          convertShapeToDims(outputShape));
  }
  else
  {
    LOG(ERROR) << "Unsupported data type";
    return false;
  }
  return true;
}
