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
#ifndef __NEURUN_IR_OPERAND_INFO_H__
#define __NEURUN_IR_OPERAND_INFO_H__

#include "ir/Shape.h"
#include "ir/TypeInfo.h"
#include "ir/Layout.h"

namespace neurun
{
namespace ir
{

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
   */
  OperandInfo(const Shape &shape, const TypeInfo &typeInfo) : _shape(shape), _typeInfo(typeInfo)
  {
    // DO NOTHING
  }
  /**
   * @brief     Construct a new OperandInfo object
   * @param[in] origin info for copy
   */
  OperandInfo(const OperandInfo &origin) : _shape(origin.shape()), _typeInfo(origin.typeInfo())
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Return tensor shape
   * @return  Tensor shape
   */
  const Shape &shape() const { return _shape; }
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

private:
  Shape _shape;
  TypeInfo _typeInfo;
};

} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_OPERAND_INFO_H__
