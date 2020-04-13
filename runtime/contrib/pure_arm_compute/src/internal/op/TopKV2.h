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
 * @file TopKV2.h
 * @brief This file contains accept function and params for TopKV2 operation
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_OP_TOPKV2_H__
#define __INTERNAL_OP_TOPKV2_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace TopKV2
{

/**
 * @brief Struct of TopKV2 operation's param
 */
struct Param
{
  int32_t outputValues_index;  /**< Output values index */
  int32_t outputIndices_index; /**< Output indices index */

  int32_t inputData_index; /**< Input data index */
  int32_t k_index;         /**< K value index */

  /**
   * @brief Construct a new Param object for TopKV2 as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for TopKV2 with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for TopKV2
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for TopKV2 with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for TopKV2
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for TopKV2
   * @return Parameters of TopKV2
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for TopKV2
   * @param [in] v Node visitor for invoking visit function of TopKV2
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace TopKV2
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_TOPKV2_H__
