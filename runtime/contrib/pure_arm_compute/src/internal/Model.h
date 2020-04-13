/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file Model.h
 * @brief This file contains classes for handle internal Model object
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_MODEL_H__
#define __INTERNAL_MODEL_H__

namespace internal
{
namespace tflite
{
namespace operand
{

/**
 * @brief Class to express index of operand.
 */
class Index
{
public:
  /**
   * @brief Construct a new Index object for operand with param.
   * @param [in] value The number of index
   */
  explicit Index(int value) : _value{value}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get index value as int
   * @return Index value as int
   */
  int asInt(void) const { return _value; }

private:
  int _value;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <vector>
#include <cstdint>

#include "misc/feature/Shape.h"
#include "misc/matrix/Shape.h"
#include "misc/kernel/Shape.h"
#include "misc/tensor/Shape.h"

namespace internal
{
namespace tflite
{
namespace operand
{

/**
 * @brief Class to express shape of operand.
 */
struct Shape : public nnfw::misc::tensor::Shape
{
public:
  /**
   * @brief Construct a new Shape object for operand with param.
   * @param [in] rank The rank value of shape
   */
  Shape(uint32_t rank);

public:
  /**
   * @brief Get dimension value of tensor as vector
   * @return Dimension value(int32_t) of tensor as vector
   */
  int32_t asVector(void) const;
  /**
   * @brief Get dimension values of tensor as feature::Shape
   * @return Dimension values of tensor as feature::Shape
   */
  nnfw::misc::feature::Shape asFeature(void) const;
  /**
   * @brief Get dimension values of tensor as matrix::Shape
   * @return Dimension values of tensor as matrix::Shape
   */
  nnfw::misc::matrix::Shape asMatrix(void) const;
  /**
   * @brief Get dimension values of tensor as kernel::Shape
   * @return Dimension values of tensor as kernel::Shape
   */
  nnfw::misc::kernel::Shape asKernel(void) const;
  /**
   * @brief Get dimension values of tensor::Shape
   * @return Dimension values of tensor::Shape
   */
  nnfw::misc::tensor::Shape asTensor(void) const;

public:
  /**
   * @brief Extend rank of Shape object for operand with param.
   * @param [in] to_rank The rank value to be extended to
   * @return N/A
   */
  void extendRank(size_t);
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <algorithm>

namespace internal
{
namespace tflite
{
namespace operand
{

/**
 * @brief Class to have data of operand.
 */
struct Data
{
  /**
   * @brief Destruct this object
   */
  virtual ~Data() = default;

  /**
   * @brief Get size of data
   * @return size of data
   */
  virtual size_t size(void) const = 0;
  /**
   * @brief Get the base address of data
   * @return the base address of data
   */
  virtual const uint8_t *base(void) const = 0;
};

/**
 * @brief Class to have cached data of operand.
 */
class CachedData final : public Data
{
public:
  /**
   * @brief Construct a new CachedData object for operand with param.
   * @param [in] base the base address of data
   * @param [in] size the size of data
   */
  CachedData(const uint8_t *base, size_t size) : _base{new uint8_t[size]}, _size{size}
  {
    std::copy(base, base + size, _base);
  }

public:
  /**
   * @brief Destruct this object
   */
  ~CachedData() { delete[] _base; }

public:
  /**
   * @brief Get size of data
   * @return size of data
   */
  size_t size(void) const override { return _size; }
  /**
   * @brief Get the base address of data
   * @return the base address of data
   */
  const uint8_t *base(void) const override { return _base; }

private:
  uint8_t *_base;
  size_t _size;
};

/**
 * @brief Class to have external data of operand.
 */
class ExternalData final : public Data
{
public:
  /**
   * @brief Construct a new ExternalData object for operand with param.
   * @param [in] base the base address of data
   * @param [in] size the size of data
   */
  ExternalData(const uint8_t *base, size_t size) : _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get size of data
   * @return size of data
   */
  size_t size(void) const override { return _size; }
  /**
   * @brief Get the base address of data
   * @return the base address of data
   */
  const uint8_t *base(void) const override { return _base; }

private:
  const uint8_t *_base;
  const size_t _size;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <memory>
#include <cassert>
#include <functional>
#include "internal/Swizzle.h"

namespace internal
{
namespace tflite
{
namespace operand
{

/**
 * @brief Class to express operand as object.
 */
class Object
{
public:
  /**
   * @brief Construct a new Object object for operand with param.
   * @param [in] shape shape of operand
   * @param [in] type type of operand
   * @param [in] scale scale of operand
   * @param [in] zeroPoint zeroPoint of operand
   */
  explicit Object(const Shape &shape, const int32_t type, const float scale,
                  const int32_t zeroPoint)
      : _shape{shape}, _type{type}, _scale{scale}, _zeroPoint{zeroPoint}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get shape of operand
   * @return Reference of shape of operand
   */
  const Shape &shape(void) const { return _shape; }
  /**
   * @brief Get type of operand
   * @return type of operand
   */
  const int32_t type(void) const { return _type; }
  /**
   * @brief Get scale of operand
   * @return scale of operand
   */
  const float scale(void) const { return _scale; }
  /**
   * @brief Get zeroPoint of operand
   * @return zeroPoint of operand
   */
  const int32_t zeroPoint(void) const { return _zeroPoint; }

private:
  void data(std::unique_ptr<Data> &&data) { _data = std::move(data); }

public:
  /**
   * @brief Get data of operand
   * @return Reference of data of operand
   */
  const Data &data(void) const { return *_data; }
  /**
   * @brief Get true if Object has data, otherwise @c false
   * @return @c true if Object has data, otherwise @c false
   */
  bool hasData(void) const { return _data != nullptr; }

public:
  /**
   * @brief Set data of operand with param
   * @param [in] args arguments of data to be set
   * @return N/A
   */
  template <typename T, typename... Args> void data(Args &&... args)
  {
    data(std::unique_ptr<T>(new T{std::forward<Args>(args)...}));
  }

public:
  /**
   * @brief Get value of data as scalar
   * @return value of data as scalar
   */
  template <typename T> T asScalar(void) const
  {
    assert((_shape.rank() == 0) || ((_shape.rank() == 1) && (_shape.dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return *(reinterpret_cast<const T *>(_data->base()));
  }

public:
  /**
   * @brief Get value of data as ReorderBits
   * @param [in] numOfBits The number of bits to be reordered to
   * @return value of data as ReorderBits
   */
  template <typename T> T asReorderBits(size_t numOfBits) const
  {
    assert((_shape.rank() == 0) || ((_shape.rank() == 1) && (_shape.dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return ReorderBits<T>(asScalar<T>(), numOfBits);
  }

private:
  const Shape _shape;
  const int32_t _type;
  const float _scale;
  const int32_t _zeroPoint;
  std::unique_ptr<Data> _data;
};

} // namespace operand
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{
namespace operand
{

/**
 * @brief Class to have object instances in a kind of set
 */
class Set
{
public:
  /**
   * @brief Iterate objects with fn
   * @param [in] fn function to be iterated
   * @return N/A
   */
  void iterate(const std::function<void(const Index &)> &fn)
  {
    for (uint32_t n = 0; n < _objects.size(); ++n)
    {
      const Index operand_index{static_cast<int>(n)};
      fn(operand_index);
    }
  }

public:
  /**
   * @brief Append Object for operand with param
   * @param [in] shape shape of operand
   * @param [in] type type of operand
   * @param [in] scale scale of operand
   * @param [in] zeroPoint zeroPoint of operand
   * @return Value of Index which has been appended to
   */
  Index append(const Shape &, int32_t type, float scale, int32_t zeroPoint);

public:
  /**
   * @brief Get Object at Index
   * @param [in] index Index to be at
   * @return Const refernece of Object
   */
  const Object &at(const Index &) const;
  /**
   * @brief Get Object at Index
   * @param [in] index Index to be at
   * @return Refernece of Object
   */
  Object &at(const Index &);
  /**
   * @brief Get size of operands in Set
   * @return Value of size
   */
  size_t size(void) const { return _objects.size(); }
  bool exist(const Index &) const;

private:
  std::vector<std::unique_ptr<Object>> _objects;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include "internal/op/NodeVisitor.h"

namespace internal
{
namespace tflite
{
namespace op
{

/**
 * @brief Class to have sequence operators.
 */
class Sequence
{
public:
  /**
   * @brief Construct a new Sequence object for operator as default
   */
  Sequence() = default;

public:
  /**
   * @brief Get size of operators in Sequence
   * @return Value of size
   */
  uint32_t size(void) const { return _ops.size(); }

public:
  /**
   * @brief Get op::Node at Index
   * @param [in] nth index to be at
   * @return Refernece of op::Node
   */
  op::Node &at(uint32_t nth) { return *(_ops.at(nth)); }
  /**
   * @brief Get op::Node at Index
   * @param [in] nth index to be at
   * @return Const refernece of op::Node
   */
  const op::Node &at(uint32_t nth) const { return *(_ops.at(nth)); }

private:
  Sequence &emplace_back(std::unique_ptr<op::Node> &&node)
  {
    _ops.emplace_back(std::move(node));
    return (*this);
  }

public:
  /**
   * @brief Add op::Node with param
   * @param [in] args arguments of op::Node to be set
   * @return Reference of Sequence
   */
  template <typename T, typename... Args> Sequence &emplace_back(Args &&... args)
  {
    return emplace_back(std::unique_ptr<T>(new T{std::forward<Args>(args)...}));
  }

private:
  std::vector<std::unique_ptr<op::Node>> _ops;
};

} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{

/**
 * @brief Class to have operand::Set as operands and op::Sequence as operators
 */
class Model
{
public:
  /**
   * @brief Get operand::Set
   * @return Reference of operand::Set
   */
  operand::Set &operands(void) { return _operands; }
  /**
   * @brief Get operand::Set
   * @return Const reference of operand::Set
   */
  const operand::Set &operands(void) const { return _operands; }

public:
  /**
   * @brief Get op::Sequence
   * @return Reference of op::Sequence
   */
  op::Sequence &operations(void) { return _operations; }
  /**
   * @brief Get op::Sequence
   * @return Const reference of op::Sequence
   */
  const op::Sequence &operations(void) const { return _operations; }

private:
  operand::Set _operands;
  op::Sequence _operations;

public:
  // TODO Hide these fields
  std::vector<operand::Index> inputs;  /**< indexes of operand as input */
  std::vector<operand::Index> outputs; /**< indexes of operand as output */
};

} // namespace tflite
} // namespace internal

#endif // __INTERNAL_MODEL_H__
