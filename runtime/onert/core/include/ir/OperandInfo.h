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

/**
 * @file  OperandInfo.h
 * @brief This file contains OperandInfo class
 */
#ifndef __ONERT_IR_OPERAND_INFO_H__
#define __ONERT_IR_OPERAND_INFO_H__

#include "ir/Index.h"
#include "ir/Layout.h"
#include "ir/Shape.h"
#include "ir/TypeInfo.h"

namespace onert
{
namespace ir
{

/**
 * @brief enum class indicating when the memory for a tensor is allocated
 */
enum class MemAllocType
{
  /**
   * @brief At compile time, shape for a tensor is known, thus requried memory capacity can be
   *        calculated
   */
  STATIC,

  /**
   * @brief At kernel execution time, shape for a tensor is known, thus requried memory capacity
   *        can be calculated
   */
  DYNAMIC
};

/**
 * @brief Class to save tensor's shape and type
 */
class OperandInfo
{
public:
  /**
   * @brief Construct a new OperandInfo object (deleted)
   */
  OperandInfo() = delete;

  /**
   * @brief     Construct a new OperandInfo object
   * @param[in] shape     Tensor shape
   * @param[in] typeInfo  Tensor data type
   * @param[in] alloc_type  When the thesor needs memory allocation
   */
  OperandInfo(const Shape &shape, const TypeInfo &typeInfo, MemAllocType alloc_type,
              bool is_const = false, bool is_variable = false, OriginIndex origin = OriginIndex())
    : _shape(shape), _typeInfo(typeInfo), _alloc_type(alloc_type), _const(is_const),
      _variable(is_variable), _origin(origin)
  {
    // DO NOTHING
  }
  /**
   * @brief     Construct a new OperandInfo object
   * @param[in] origin info for copy
   */
  OperandInfo(const OperandInfo &origin) = default;

  /**
   * @brief Create a static OperandInfo object
   */
  static OperandInfo createStaticInfo(const Shape &shape, const TypeInfo &typeInfo)
  {
    return OperandInfo(shape, typeInfo, MemAllocType::STATIC);
  }

public:
  /**
   * @brief   Return tensor shape
   * @return  Tensor shape
   */
  const Shape &shape() const { return _shape; }
  /**
   * @brief   Return mutable tensor shape
   * @return  Tensor shape
   */
  Shape &shape() { return _shape; }
  /**
   * @brief set shape
   */
  void shape(const ir::Shape &new_shape) { _shape = new_shape; }
  /**
   * @brief   Return tensor data type info
   * @return  Tensor data type
   */
  const TypeInfo &typeInfo() const { return _typeInfo; }
  /**
   * @brief     Set type information
   * @param[in] typeInfo Type information
   */
  void typeInfo(const ir::TypeInfo &typeInfo) { _typeInfo = typeInfo; }
  /**
   * @brief   Set tensor data type
   */
  void type(const DataType type) { _typeInfo.type(type); }
  /**
   * @brief   Return size of tensor (bytes)
   * @return  Tensor size
   */
  size_t total_size() const { return _shape.num_elements() * sizeOfDataType(_typeInfo.type()); }

  MemAllocType memAllocType() const { return _alloc_type; }
  void setAsConstant() { _const = true; }
  void setAsNonConst() { _const = false; }
  bool isConstant() const
  {
    // Impossible case: constant and dynamic operand
    assert(!(isDynamic() && _const));
    return _const;
  }
  void setAsVariable()
  {
    // Impossible case: constant or dynamic operand
    // The variable operand with buffer is not supported yet
    assert(!(isDynamic() || _const));
    _variable = true;
  }
  bool isVariable() const { return _variable; }
  bool isDynamic() const { return _alloc_type == MemAllocType::DYNAMIC; }
  void setDynamic() { _alloc_type = MemAllocType::DYNAMIC; }
  OriginIndex originIndex() const { return _origin; }
  void setOriginIndex(OriginIndex origin) { _origin = origin; }

private:
  Shape _shape;
  TypeInfo _typeInfo;
  MemAllocType _alloc_type;
  bool _const;
  bool _variable;
  OriginIndex _origin;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERAND_INFO_H__
