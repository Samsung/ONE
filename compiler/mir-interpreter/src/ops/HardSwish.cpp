/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "HardSwish.h"
#include "Common.h"

namespace mir_interpreter
{

template <typename T> struct HardSwishImpl
{
  static void run(const mir::TensorVariant &input, mir::TensorVariant &result);
};

template <typename T>
void HardSwishImpl<T>::run(const mir::TensorVariant &input, mir::TensorVariant &result)
{
  auto output_data = reinterpret_cast<T *>(result.atOffset(0));
  auto input_data = reinterpret_cast<T *>(input.atOffset(0));
  auto in_end = input_data + input.getShape().numElements();
  for (; input_data < in_end; input_data++, output_data++)
  {
    const auto in = *input_data;
    *output_data = in * std::min<T>(6.f, std::max<T>(0.f, in + 3.f)) / 6.f;
  }
}

template <> struct HardSwishImpl<uint8_t>
{
  static void run(const mir::TensorVariant &input, mir::TensorVariant &result)
  {
    throw std::runtime_error{"NYI"};
  }
};

void HardSwish(const mir::TensorVariant &input, mir::TensorVariant &result)
{
  dispatch<HardSwishImpl>(input.getElementType(), input, result);
}

} // namespace mir_interpreter
