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

#ifndef __TFINFO_V2_TENSORSIGNATURE_H__
#define __TFINFO_V2_TENSORSIGNATURE_H__

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

namespace tfinfo
{
inline namespace v2
{

/**
 * @brief Supported Data Types
 */
enum class DataType
{
  UNKNOWN,

  FLOAT32, // IEEE 32-bit floating point
  /* To be added */
};

/**
 * @brief Class to represent axis and size of dims.
 *        User should enter axis and size of dim(s) when input tensor(s) contain(s) unknown dim(s).
 *        Such axis and size of dim(s) will be stored in ShapeHint.
 */
class ShapeHint
{
  using AxisHint = uint32_t;
  using SizeHint = uint64_t;

public:
  ShapeHint() = default;

  void add(AxisHint axis, SizeHint size)
  {
    if (_dims.find(axis) != _dims.end())
      throw std::runtime_error("dim value already exists");

    _dims[axis] = size;
  }

  std::map<AxisHint, SizeHint>::const_iterator cbegin() const { return _dims.cbegin(); }

  std::map<AxisHint, SizeHint>::const_iterator cend() const { return _dims.cend(); }

  bool empty() { return _dims.size() == 0; }

  size_t size() { return _dims.size(); }

private:
  std::map<AxisHint, SizeHint> _dims;
};

using TensorName = std::string;

/**
 * @brief Class to store input and output tensor information
 */
class TensorSignature final
{
public:
  enum class Kind
  {
    Input,
    Output
  };

  TensorSignature(const Kind kind, const std::string &name) : _kind(kind), _tensor_name()
  {
    // tensor name can be a form of "placeholder:0" or "placeholder".
    // If tensor index is omitted, ":0" is appended
    auto pos = name.find(":");
    if (pos == std::string::npos)
      _tensor_name.assign(name + ":0");
    else
      _tensor_name.assign(name);
  }

  TensorSignature(const Kind kind, const std::string &name, const ShapeHint &shape_hint)
    : TensorSignature(kind, name)
  {
    _shape_hint = shape_hint;
  }

public:
  Kind kind() const { return _kind; }

  const TensorName &name() { return _tensor_name; }

  ShapeHint &shapeHint() { return _shape_hint; }

private:
  Kind _kind;
  std::string _tensor_name;
  ShapeHint _shape_hint;
};

using TensorSignatures = std::vector<std::unique_ptr<TensorSignature>>;

} // namespace v2
} // namespace tfinfo

#endif // __TFINFO_V2_TENSORSIGNATURE_H__
