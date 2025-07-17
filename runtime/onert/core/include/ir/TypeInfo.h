/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_TYPEINFO_H__
#define __ONERT_IR_TYPEINFO_H__

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "ir/DataType.h"
#include "ir/Sparsity.h"

namespace onert::ir
{

/**
 * @brief Struct to hold quantization information of a tensor
 *
 * @note  This struct is used only for quantized tensors.
 *
 *        For non-quantized tensors, scales, and zero_points have only one element,
 *        and scale value is zero.
 *
 *        For per-channel quantized tensors, scales vectors should have same size as number of
 *        channels of the tensor. And zero_points vector should be same size as scales vector if
 *        quantization type is assymetric, otherwise it will be ignored.
 */
struct Quantization
{
  std::vector<float> scales = {0.0f};
  std::vector<int32_t> zero_points = {0};
};

class TypeInfo
{
public:
  TypeInfo() = delete;

  explicit TypeInfo(DataType type) : _type{type}, _sparsity{nullptr} {}

  TypeInfo(DataType type, float scale, int32_t zero_point) : _type{type}, _sparsity{nullptr}
  {
    quantization(scale, zero_point);
  }

public:
  DataType type() const { return _type; }
  float scale() const { return _quant.scales[0]; }
  const std::vector<float> &scales() const { return _quant.scales; }
  int32_t zero_point() const { return _quant.zero_points[0]; }
  const std::vector<int32_t> &zero_points() const { return _quant.zero_points; }
  const ir::Sparsity *sparsity() const { return _sparsity.get(); }
  void quantization(float scale, int32_t zero_point)
  {
    assert(requireQuantParam(_type) || scale == 0); // Quantize param type, or scale = 0
    _quant.scales = {scale};
    _quant.zero_points = {zero_point};
  }
  void quantization(std::vector<float> &&scales, std::vector<int32_t> &&zero_points)
  {
    assert(requireQuantParam(_type)); // Only quantize param types can call this
    assert(scales.size() != 0);       // Not allow empty scales vector when this is called
    assert(zero_points.size() == 0 ||
           zero_points.size() == scales.size()); // Symmetric or Asymmetric
    // Not allow meaningless quantization parameters
    assert(scales.size() == 1 ||
           std::all_of(scales.begin(), scales.end(), [](float s) { return s != 0; }));

    _quant.scales = std::move(scales);
    _quant.zero_points = std::move(zero_points);
  }
  void sparsity(std::shared_ptr<ir::Sparsity> &&sparsity) { _sparsity = std::move(sparsity); }

public:
  void type(const DataType type) { _type = type; }

private:
  DataType _type;
  ir::Quantization _quant;
  std::shared_ptr<ir::Sparsity> _sparsity;
};

bool operator==(const TypeInfo &lhs, const TypeInfo &rhs);
bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs);

} // namespace onert::ir

#endif // __ONERT_IR_TYPEINFO_H__
