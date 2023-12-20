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

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "ir/DataType.h"
#include "ir/Sparsity.h"

namespace onert
{
namespace ir
{

struct Quantization
{
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
};

class TypeInfo
{
public:
  TypeInfo() = delete;

  explicit TypeInfo(DataType type) : _type{type}, _sparsity{nullptr} { quantization(0.f, 0); }

  TypeInfo(DataType type, float scale, int32_t zero_point) : _type{type}, _sparsity{nullptr}
  {
    quantization(scale, zero_point);
  }

public:
  DataType type() const { return _type; }
  float scale() const { return _quant.scales[0]; }
  const std::vector<float> &scales() const { return _quant.scales; }
  int32_t zero_point() const
  {
    assert(_quant.zero_points.size() == 1);
    return _quant.zero_points[0];
  }
  const std::vector<int32_t> &zero_points() const { return _quant.zero_points; }
  const ir::Sparsity *sparsity() const { return _sparsity.get(); }
  void quantization(float scale, int32_t zero_point)
  {
    _quant.scales.resize(1);
    _quant.scales[0] = scale;
    _quant.zero_points.resize(1);
    _quant.zero_points[0] = zero_point;
  }
  void quantization(std::vector<float> &&scales, std::vector<int32_t> &&zero_points)
  {
    _quant.scales = scales;
    _quant.zero_points = zero_points;
  }
  void sparsity(std::shared_ptr<ir::Sparsity> sparsity) { _sparsity = sparsity; }

public:
  void type(const DataType type) { _type = type; }

private:
  DataType _type;
  ir::Quantization _quant;
  std::shared_ptr<ir::Sparsity> _sparsity;
};

bool operator==(const TypeInfo &lhs, const TypeInfo &rhs);
bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TYPEINFO_H__
