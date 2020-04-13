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
 * @file Node.h
 * @brief This file contains struct of Node and NodeVisitor
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_OP_NODE_H__
#define __INTERNAL_OP_NODE_H__

namespace internal
{
namespace tflite
{
namespace op
{

/**
 * @brief Struct of operation NodeVisitor
 */
struct NodeVisitor;

/**
 * @brief Struct of operation Node
 */
struct Node
{
  /**
   * @brief Destroy the Node object for operation
   */
  virtual ~Node() = default;

  /**
   * @brief Function for accepting node for operation
   * @param [in] v Node visitor for invoking visit function of operation
   * @return N/A
   */
  virtual void accept(NodeVisitor &&) const = 0;
};

} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_NODE_H__
