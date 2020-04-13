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
 * @file Gather.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file defines Gather operation
 */

#ifndef __INTERNAL_OP_GATHER_H__
#define __INTERNAL_OP_GATHER_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Gather
{

/**
 * @brief Struct to manipulate parameter for Gather operation
 */
struct Param
{
  int32_t ofm_index; //!< index for output feature map

  int32_t ifm_index;     //!< index for ifm tensor
  int32_t indices_index; //!< index for indices tensor
  int32_t axis_index;    //!< index for axis

  /**
   * @brief Default Constructor
   */
  Param() = default;
  /**
   * @brief Construct a new Param object
   * @param[in] inputCount the number of inputs
   * @param[in] inputs pointer for input data
   * @param[in] outputCount the number of outputs
   * @param[in] outputs pointer for input data
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define Gather Operation
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Tanh Node object
   * @param param Parameter for Tanh Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Default Destructor
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameter
   * @return Param reference
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Accept a NodeVisitor so that it can visit this node
   * @param [in] v Visitor
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param; //!< parameter for Gather node
};

} // namespace Gather
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_GATHER_H__
