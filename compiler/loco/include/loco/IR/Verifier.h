/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_VERIFIER_H__
#define __LOCO_IR_VERIFIER_H__

#include "loco/IR/Graph.h"

#include <memory>

namespace loco
{

/**
 * @brief Possible error categories
 *
 * This enum class enumerates all the possible validation failure reasons.
 *
 * WARN DO NOT serialize this code. The tag value is subject to change.
 */
enum class ErrorCategory
{
  MissingArgument,
  /* TO BE ADDED */
};

/**
 * @brief The details of each error
 */
template <ErrorCategory Code> class ErrorDetail;

/**
 * @brief The details of MissingArgument error
 */
template <> class ErrorDetail<ErrorCategory::MissingArgument>
{
public:
  ErrorDetail(loco::Node *node, uint32_t index) : _node{node}, _index{index}
  {
    // DO NOTHING
  }

public:
  /// @brief The node with missing arguments
  loco::Node *node(void) const { return _node; }
  /// @brief The missing argument index
  uint32_t index(void) const { return _index; }

private:
  loco::Node *_node;
  uint32_t _index;
};

/**
 * @brief Error listener interface
 *
 * DOo NOT inherit this interface. Use DefaultErrorListener instead.
 */
struct IErrorListener
{
  virtual ~IErrorListener() = default;

  virtual void notify(const ErrorDetail<ErrorCategory::MissingArgument> &) = 0;
};

/**
 * @brief Error listener (with default implementation)
 */
struct ErrorListener : public IErrorListener
{
  virtual ~ErrorListener() = default;

  void notify(const ErrorDetail<ErrorCategory::MissingArgument> &) override { return; }
};

/**
 * @brief Validate a loco graph
 *
 * "valid" returns true if a given graph has no error.
 *
 * NOTE Given a valid(non-null) listener, "valid" notifies error details to the listener.
 */
bool valid(Graph *g, std::unique_ptr<ErrorListener> &&l = nullptr);

} // namespace loco

#endif // __LOCO_IR_VERIFIER_H__
