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
 * @file MaxPool2D.h
 * @brief This file contains accept function and params for MaxPool2D operation
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_OP_MAX_POOL_2D_H__
#define __INTERNAL_OP_MAX_POOL_2D_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace MaxPool2D
{
namespace Explicit
{

/**
 * @brief Struct of MaxPool2D(Explicit) operation's param
 */
struct Param
{
  int32_t ofm_index; /**< Output format index */

  int32_t ifm_index; /**< Input format index */

  int32_t kw_index; /**< Kernel width index */
  int32_t kh_index; /**< Kernel height index */

  int32_t hstride_index; /**< Horizontal stride index */
  int32_t vstride_index; /**< Vertical stride index */

  int32_t padding_left_index;   /**< Left padding index */
  int32_t padding_right_index;  /**< Right padding index */
  int32_t padding_top_index;    /**< Top padding index */
  int32_t padding_bottom_index; /**< Bottom padding index */

  int32_t activation_index; /**< Activation index */

  /**
   * @brief Construct a new Param object for MaxPool2D(Explicit) as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for MaxPool2D(Explicit) with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for MaxPool2D(Explicit)
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for MaxPool2D(Explicit) with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for MaxPool2D(Explicit)
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for MaxPool2D(Explicit)
   * @return Parameters of MaxPool2D(Explicit)
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for MaxPool2D(Explicit)
   * @param [in] v Node visitor for invoking visit function of MaxPool2D(Explicit)
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace Explicit

namespace Implicit
{

/**
 * @brief Struct of MaxPool2D(Implicit) operation's param
 */
struct Param
{
  int32_t ofm_index; /**< Output format index */

  int32_t ifm_index; /**< Input format index */

  int32_t kw_index; /**< Kernel width index */
  int32_t kh_index; /**< Kernel height index */

  int32_t hstride_index; /**< Horizontal stride index */
  int32_t vstride_index; /**< Vertical stride index */

  int32_t padding_index;    /**< Padding index */
  int32_t activation_index; /**< Activation index */

  /**
   * @brief Construct a new Param object for MaxPool2D(Implicit) as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for MaxPool2D(Implicit) with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for MaxPool2D(Implicit)
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for MaxPool2D(Implicit) with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for MaxPool2D(Implicit)
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for MaxPool2D(Implicit)
   * @return Parameters of MaxPool2D(Implicit)
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for MaxPool2D(Implicit)
   * @param [in] v Node visitor for invoking visit function of MaxPool2D(Implicit)
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace Implicit
} // namespace MaxPool2D
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_MAX_POOL_2D_H__
