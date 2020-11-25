/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_INTERNAL_TYPE_H__
#define __ONERT_IR_INTERNAL_TYPE_H__

#include <cstdint>

namespace onert
{
namespace ir
{

enum class Activation
{
  NONE = 0,
  RELU = 1,
  RELU1 = 2,
  RELU6 = 3,
  TANH = 4,
  SIGMOID = 5
};

struct Stride
{
  uint32_t vertical;
  uint32_t horizontal;
};

struct Dilation
{
  uint32_t width_factor;
  uint32_t height_factor;
};

enum class FullyConnectedWeightsFormat
{
  Default = 0,
  Shuffled4x16Int8 = 1,
  Shuffled16x1Float32 = 127
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_INTERNAL_TYPE_H__
