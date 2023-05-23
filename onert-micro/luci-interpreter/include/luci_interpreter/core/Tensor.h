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

#ifndef LUCI_INTERPRETER_CORE_TENSOR_H
#define LUCI_INTERPRETER_CORE_TENSOR_H

#include "luci_interpreter/core/DataType.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luci_interpreter
{

class Tensor
{
public:
#ifndef DIS_QUANT
  static float scale(const circle::Tensor *circle_tensor)
  {
    const auto *quant_params = circle_tensor->quantization();
    if (quant_params == nullptr)
    {
      assert(false && "There is no quantization params");
      return 0;
    }

    return *quant_params->scale()->cbegin();
  }

  static int32_t zero_point(const circle::Tensor *circle_tensor)
  {
    const auto *quant_params = circle_tensor->quantization();
    if (quant_params == nullptr)
    {
      assert(false && "There is no quantization params");
      return 0;
    }

    return *quant_params->zero_point()->cbegin();
  }

  static const std::vector<float> scales(const circle::Tensor *circle_tensor)
  {
    const auto *quant_params = circle_tensor->quantization();
    if (quant_params == nullptr)
    {
      assert(false && "There is no quantization params");
      return {};
    }
    assert(quant_params->scale() != nullptr);
    std::vector<float> scales(quant_params->scale()->cbegin(), quant_params->scale()->cend());

    return scales;
  }

  static const std::vector<int32_t> zero_points(const circle::Tensor *circle_tensor)
  {
    const auto *quant_params = circle_tensor->quantization();
    if (quant_params == nullptr)
    {
      assert(false && "There is no quantization params");
      return {};
    }
    assert(quant_params->zero_point() != nullptr);
    std::vector<int32_t> zero_points(quant_params->zero_point()->cbegin(),
                                     quant_params->zero_point()->cend());

    return zero_points;
  }

  static int32_t quantized_dimension(const circle::Tensor *circle_tensor)
  {
    const auto *quant_params = circle_tensor->quantization();
    if (quant_params == nullptr)
    {
      assert(false && "There is no quantization params");
      return 0;
    }
    return quant_params->quantized_dimension();
  }
#endif

  static bool is_constant_tensor(const luci_interpreter::CircleReader *reader,
                                 const circle::Tensor *circle_tensor)
  {
    return reader->buffers()[circle_tensor->buffer()]->data() != nullptr;
  }

  static DataType element_type(const circle::Tensor *circle_tensor)
  {
    return luci_datatype(circle_tensor->type());
  }

  static VectorWrapper<int32_t> tensor_shape(const circle::Tensor *circle_tensor)
  {
    return wrap(circle_tensor->shape());
  }

  static int num_dims(const circle::Tensor *circle_tensor)
  {
    // TODO check removing of wrap
    auto const const_dims = wrap(circle_tensor->shape());
    return const_dims.size();
  }

  static int32_t dim(const circle::Tensor *circle_tensor, int i)
  {
    // TODO check removing of wrap
    assert(i >= 0);
    auto const const_dims = wrap(circle_tensor->shape());
    assert(i < const_dims.size());

    return const_dims[i];
  }

  static int32_t num_elements(const circle::Tensor *circle_tensor)
  {
    int32_t result = 1;
    auto const const_dims = wrap(circle_tensor->shape());
    for (const int32_t dim : const_dims)
    {
      result *= dim;
    }
    return result;
  }
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_TENSOR_H
