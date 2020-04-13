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
 * @file Conv2D.h
 * @brief This file contains accept function and params for Conv2D operation
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_OP_CONV_2D_H__
#define __INTERNAL_OP_CONV_2D_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace Conv2D
{
namespace Explicit
{

/**
 * @brief Struct of Conv2D(explicit) operation's param
 */
struct Param
{
  int32_t ofm_index; /**< Output format index */

  int32_t ifm_index;  /**< Input format index */
  int32_t ker_index;  /**< Kernel index */
  int32_t bias_index; /**< Bias index */

  int32_t hstride_index; /**< Horizontal stride index */
  int32_t vstride_index; /**< Vertical stride index */

  int32_t padding_left_index;   /**< Left padding index */
  int32_t padding_right_index;  /**< Right padding index */
  int32_t padding_top_index;    /**< Top padding index */
  int32_t padding_bottom_index; /**< Bottomd padding index */

  int32_t activation_index; /**< Activation index */

  /**
   * @brief Construct a new Param object for Conv2D(explicit) as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for Conv2D(explicit) with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for Conv2D(explicit)
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for conv2D(explicit) with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for conv2D(explicit)
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for conv2D(explicit)
   * @return Parameters of conv2D(explicit)
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for conv2D(explicit)
   * @param [in] v Node visitor for invoking visit function of conv2D(explicit)
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
 * @brief Struct of Conv2D(implicit) operation's param
 */
struct Param
{
  int32_t ofm_index; /**< Output format index */

  int32_t ifm_index;  /**< Input format index */
  int32_t ker_index;  /**< Kernel index */
  int32_t bias_index; /**< Bias index */

  int32_t hstride_index; /**< Horizontal stride index */
  int32_t vstride_index; /**< Vertical stride index */

  int32_t padding_index;    /**< Padding index */
  int32_t activation_index; /**< Activation index */

  /**
   * @brief Construct a new Param object for Conv2D(implicit) as default
   */
  Param() = default;

  /**
   * @brief Construct a new Param object for Conv2D(implicit) with params
   * @param [in] inputCount The number of input
   * @param [in] inputs Array containing inputs
   * @param [in] outputCount The number of output
   * @param [in] outputs Array containing outputs
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define operation node for Conv2D(implicit)
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new Node object for conv2D(implicit) with param
   * @param [in] param Parameters for Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Destroy the Node object for conv2D(implicit)
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameters for conv2D(implicit)
   * @return Parameters of conv2D(implicit)
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Function for accepting node for conv2D(implicit)
   * @param [in] v Node visitor for invoking visit function of conv2D(implicit)
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param;
};

} // namespace Implicit
} // namespace Conv2D
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_CONV_2D_H__
