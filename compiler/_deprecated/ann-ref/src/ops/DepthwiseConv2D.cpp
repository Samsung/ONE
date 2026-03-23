/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "DepthwiseConv2D.h"
#include "Assert.h"

#include "internal/Spatial.h"

bool depthwiseConvPrepare(const Shape &input, const Shape &filter, const Shape &bias,
                          int32_t padding_left, int32_t padding_right, int32_t padding_top,
                          int32_t padding_bottom, int32_t stride_width, int32_t stride_height,
                          Shape *output)
{
  ASSERT(input.type == filter.type);
  if (input.type == OperandType::TENSOR_QUANT8_ASYMM)
  {
    ASSERT(bias.type == OperandType::TENSOR_INT32);
  }
  else
  {
    ASSERT(input.type == bias.type);
  }
  ASSERT(getNumberOfDimensions(input) == 4);
  ASSERT(getNumberOfDimensions(filter) == 4);
  ASSERT(getNumberOfDimensions(bias) == 1);

  ASSERT(getSizeOfDimension(filter, 3) == getSizeOfDimension(bias, 0));

  uint32_t channels_out = getSizeOfDimension(filter, 3);
  uint32_t width = getSizeOfDimension(input, 2);
  uint32_t height = getSizeOfDimension(input, 1);
  uint32_t filterWidth = getSizeOfDimension(filter, 2);
  uint32_t filterHeight = getSizeOfDimension(filter, 1);
  uint32_t batches = getSizeOfDimension(input, 0);

  uint32_t outWidth = computeOutSize(width, filterWidth, stride_width, padding_left, padding_right);
  uint32_t outHeight =
      computeOutSize(height, filterHeight, stride_height, padding_top, padding_bottom);

  output->type = input.type;
  output->dimensions = {batches, outHeight, outWidth, channels_out};
  return true;
}
