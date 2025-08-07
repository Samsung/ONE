/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef ONERT_MICRO_CORE_QUANTIZATION_DATA_H
#define ONERT_MICRO_CORE_QUANTIZATION_DATA_H

#include "core/reader/OMCircleReader.h"

#include <cmath>
#include <cstddef>

namespace onert_micro
{
namespace core
{

// clang-format off

// ------------------------------------------------------------------------------------------------

template <typename T>
class OMQuantizationData
{
public:
  enum QuantizationType
  {
    CWQ,
    LWQ
  };

private:
  T *_data = nullptr;
  const circle::QuantizationParameters *_params = nullptr;
  QuantizationType _type = LWQ;

public:
  OMQuantizationData(T *data, const circle::Tensor *tensor)
    : _data(data)
  {
    assert(data != nullptr);
    assert(tensor != nullptr);
    assert(tensor->quantization() != nullptr);

    _params = tensor->quantization();
    
    if (Scales().Length() > 1)
    {
      _type = CWQ;
    }
  }

public:
  const flatbuffers::Vector<float> &Scales() const
  {
    return *(_params->scale());
  }

  const flatbuffers::Vector<int64_t> &ZeroPoint() const
  {
    return *(_params->zero_point());
  }

public:
  int64_t ZeroPointAt(size_t idx) const
  {
    return ZeroPoint()[idx];
  }

  float ScaleAt(size_t idx) const
  {
    return Scales()[idx];
  }

public:
  float Dequantize(T qvalue)
  {
    float result = (qvalue - ZeroPointAt(0)) * ScaleAt(0);
    return result;
  }

  T Quantize(float value)
  {
    float fvalue = value / ScaleAt(0) + ZeroPointAt(0);
    T qvalue = Clamp(std::round(fvalue));
    return qvalue;
  }

public:
  float DataAt(size_t idx)
  {
    return Dequantize(_data[idx]);
  }

  void SetDataAt(size_t idx, float value)
  {
    _data[idx] = Quantize(value);
  }

private:
  T Clamp(float value)
  {
    using limits = std::numeric_limits<T>;

    constexpr static auto kMin = static_cast<float>(limits::min());
    constexpr static auto kMax = static_cast<float>(limits::max());

    value = std::max(value, kMin);
    value = std::min(value, kMax);

    return static_cast<T>(value);
  }
};

// ------------------------------------------------------------------------------------------------

} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_QUANTIZATION_DATA_H
