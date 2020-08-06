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

#include <cstdint>
#include <vector>

#include "ir/DataType.h"

namespace onert
{
namespace ir
{

class TypeInfo
{
public:
  TypeInfo() = delete;

  explicit TypeInfo(DataType type, float scale = 0, int32_t offset = 0)
      : _type(type), _scale(scale), _offset(offset), _sparse(false)
  {
  }

public:
  DataType type() const { return _type; }
  float scale() const { return _scale; }
  int32_t offset() const { return _offset; }
  bool sparse() const { return _sparse; }
  const uint16_t *w1_segments() const { return _w1_segments.data(); }
  const uint16_t *w1_indices() const { return _w1_indices.data(); }

public:
  void type(const DataType type) { _type = type; }
  void sparse2DMetadata(std::vector<uint16_t> &&w1_segments, std::vector<uint16_t> &&w1_indices)
  {
    _sparse = true;
    _w1_segments = w1_segments;
    _w1_indices = w1_indices;
  }

private:
  DataType _type;
  // for quantization
  float _scale;
  int32_t _offset;
  // for sparsity
  bool _sparse;
  std::vector<uint16_t> _w1_segments;
  std::vector<uint16_t> _w1_indices;
};

bool operator==(const TypeInfo &lhs, const TypeInfo &rhs);
bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TYPEINFO_H__
