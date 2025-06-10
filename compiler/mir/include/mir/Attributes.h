/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef OP_ATTRIBUTES_H
#define OP_ATTRIBUTES_H

#include <cstdint>
#include <vector>
#include "mir/DataFormat.h"
#include "mir/ops/PaddingType.h"

namespace mir
{

struct Conv2DOpAttributes
{
  Conv2DOpAttributes() = default;

  std::vector<std::int32_t> strides{1, 1};
  std::vector<std::int32_t> padding_before{0, 0};
  std::vector<std::int32_t> padding_after{0, 0};
  std::int32_t num_groups{1};
  DataFormat data_format{DataFormat::NHWC};
};

struct AvgPool2DOpAttributes
{
  AvgPool2DOpAttributes() = default;

  std::vector<std::int32_t> window{1, 1};
  std::vector<std::int32_t> strides{1, 1};
  std::vector<std::int32_t> padding_before{0, 0};
  std::vector<std::int32_t> padding_after{0, 0};
  DataFormat data_format{DataFormat::NHWC};
  bool include_pad{true};
};

struct MaxPool2DOpAttributes
{
  MaxPool2DOpAttributes() = default;

  std::vector<std::int32_t> window{1, 1};
  std::vector<std::int32_t> strides{1, 1};
  std::vector<std::int32_t> padding_before{0, 0};
  std::vector<std::int32_t> padding_after{0, 0};
  DataFormat data_format{DataFormat::NHWC};
};

struct Deconv2DOpAttributes
{
  Deconv2DOpAttributes() = default;

  std::vector<std::int32_t> strides{1, 1};
  std::vector<std::int32_t> padding_before{0, 0};
  std::vector<std::int32_t> padding_after{0, 0};
  DataFormat data_format{DataFormat::NHWC};
  ops::PaddingType padding_type{ops::PaddingType::Explicit};
};

struct PadOpAttributes
{
  PadOpAttributes() : padding_value(0.0) {}
  PadOpAttributes(unsigned dims) : padding_before(dims), padding_after(dims), padding_value(0.0) {}

  std::vector<std::int32_t> padding_before;
  std::vector<std::int32_t> padding_after;
  float padding_value;
};
} // namespace mir

#endif
