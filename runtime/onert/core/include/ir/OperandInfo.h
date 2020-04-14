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

#include "ir/Shape.h"
#include "ir/TypeInfo.h"
#include "ir/Layout.h"

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
   *
   * @todo Deprecated this constructor because setting member var implicitly can cause bug later.
   *       Please use the third constructor. (This constor needs for now not to break previous code)
   */
  OperandInfo(const Shape &shape, const TypeInfo &typeInfo)
      : _shape(shape), _typeInfo(typeInfo), _alloc_type(MemAllocType::STATIC)
  {
    // DO NOTHING
  }
  /**
   * @brief     Construct a new OperandInfo object
   * @param[in] shape     Tensor shape
   * @param[in] typeInfo  Tensor data type
   * @param[in] alloc_type  When the thesor needs memory allocation
   */
  OperandInfo(const Shape &shape, const TypeInfo &typeInfo, MemAllocType alloc_type)
      : _shape(shape), _typeInfo(typeInfo), _alloc_type(alloc_type)
  {
    // DO NOTHING
  }
  /**
   * @brief     Construct a new OperandInfo object
   * @param[in] origin info for copy
   */
  OperandInfo(const OperandInfo &origin) = default;

public:
  /**
   * @brief   Return tensor shape
   * @return  Tensor shape
   */
  const Shape &shape() const { return _shape; }
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
   * @brief   Set tensor data type
   */
  void type(const DataType type) { _typeInfo.type(type); }
  /**
   * @brief   Return size of tensor (bytes)
   * @return  Tensor size
   */
  size_t total_size() const { return _shape.num_elements() * sizeOfDataType(_typeInfo.type()); }

  MemAllocType memAllocType() const { return _alloc_type; }
  void memAllocType(MemAllocType alloc_type) { _alloc_type = alloc_type; }

private:
  Shape _shape;
  TypeInfo _typeInfo;

  MemAllocType _alloc_type;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERAND_INFO_H__
