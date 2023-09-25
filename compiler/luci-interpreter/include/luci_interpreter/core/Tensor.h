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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luci_interpreter
{

class Shape
{
public:
  explicit Shape(int rank) : _dims(rank, 0) {}

  Shape(std::initializer_list<int32_t> dims) : _dims(dims.begin(), dims.end()) {}

  int num_dims() const { return _dims.size(); }

  int32_t dim(int i) const
  {
    assert(i >= 0 && i < static_cast<int>(_dims.size()));
    return _dims[i];
  }

  int32_t &dim(int i)
  {
    assert(i >= 0 && i < static_cast<int>(_dims.size()));
    return _dims[i];
  }

  int32_t num_elements() const
  {
    int32_t result = 1;
    for (const int32_t dim : _dims)
    {
      result *= dim;
    }
    return result;
  }

  // TODO Replace num_elements
  int64_t large_num_elements() const
  {
    int64_t result = 1;
    for (const int32_t dim : _dims)
    {
      result *= dim;
    }
    return result;
  }

  bool operator==(const Shape &other) const { return _dims == other._dims; }

  bool operator!=(const Shape &other) const { return !operator==(other); }

private:
  std::vector<int32_t> _dims;
};

// Tensor affine quantization parameters.
//
// The relationship between real and quantized values:
//   real_value = (quantized_value - zero_point) * scale
//
// In per-tensor case, 'scale' and 'zero_point' are one element each.
// In per-channel case, 'scale' and 'zero_point' are N elements each, where N is the size
// of the quantized dimension.
//
// Note that due to historical and performance reasons, per-tensor quantization uses unsigned
// integer types, while per-channel uses signed types assuming 'zero_point' == 0.
struct AffineQuantization
{
  std::vector<float> scale;
  std::vector<int32_t> zero_point;
  int32_t quantized_dimension;
};

class Tensor
{
public:
  Tensor(DataType element_type, Shape shape, AffineQuantization quantization, std::string name);

  DataType element_type() const { return _element_type; }

  const Shape &shape() const { return _shape; }

  float scale() const
  {
    assert(_quantization.scale.size() == 1);
    return _quantization.scale[0];
  }

  int32_t zero_point() const
  {
    assert(_quantization.zero_point.size() == 1);
    return _quantization.zero_point[0];
  }

  const std::vector<float> &scales() const { return _quantization.scale; }

  const std::vector<int32_t> &zero_points() const { return _quantization.zero_point; }

  int32_t quantized_dimension() const { return _quantization.quantized_dimension; }

  template <typename T> const T *data() const
  {
    static_assert(std::is_same<uint8_t, char>::value or
                  std::is_same<uint8_t, unsigned char>::value);
    return reinterpret_cast<const T *>(_data);
  }

  template <typename T> T *data()
  {
    static_assert(std::is_same<uint8_t, char>::value or
                  std::is_same<uint8_t, unsigned char>::value);
    return reinterpret_cast<T *>(_data);
  }

  const std::string &name() const { return _name; }

  void readData(void *data_ptr, size_t data_size) const;

  void writeData(const void *data_ptr, size_t data_size);

  void resize(const Shape &new_shape);

  void set_data_buffer(uint8_t *buffer)
  {
    if (buffer == nullptr)
    {
      _data_allocated = false;
    }
    else
    {
      _data_allocated = true;
    }
    _data = buffer;
  }

  bool is_observable() const { return _is_observable; }

  void set_observable(bool value) { _is_observable = value; }

  bool is_allocatable() const { return _is_allocatable; }

  void set_allocatable(bool value) { _is_allocatable = value; }

  bool is_data_allocated() const { return _data_allocated; }

  int32_t get_offset() const { return _offset; }

  void set_offset(int32_t offset) { _offset = offset; }

private:
  DataType _element_type;
  Shape _shape;
  AffineQuantization _quantization;
  uint8_t *_data = nullptr;
  std::string _name;
  bool _data_allocated = false;
  // Write of tensor is reported to registered Observers only if this tensor is observable
  // This is needed for tensors used in kernel implementation, but not present in original model.
  bool _is_observable = true;
  // Memory manager is called for tensor only if it is "allocatable".
  // Kernel configuration could disable allocation of some tensors if they are not needed for
  // particular operation.
  bool _is_allocatable = true;
  // Used by static memory manager.
  // Stores the offset from the beginning of the allocated memory buffer.
  int32_t _offset = -1;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_TENSOR_H
