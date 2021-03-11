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

#ifndef __LOCO_SERVICE_SHAPE_INFERENCE_RULE_H__
#define __LOCO_SERVICE_SHAPE_INFERENCE_RULE_H__

#include "loco/IR/Domain.h"
#include "loco/IR/Dialect.h"
#include "loco/IR/Node.h"
#include "loco/IR/NodeShape.h"

namespace loco
{

struct ShapeInferenceRule
{
  virtual ~ShapeInferenceRule() = default;

  enum class API
  {
    /**
     * API v1
     *
     * This API uses "shape_get" method to query the shape of other nodes.
     */
    V1,

    /**
     * API v2
     *
     * This API uses a given context (defined below) to query the shape of other nodes.
     */
    V2,
  };

  /// @brief Check whether a given API is available or not
  virtual bool support(const API &api) const
  {
    // To be backward compatible
    return api == API::V1;
  }

  /// @brief Return true if this rule recognizes a given dialect
  virtual bool recognize(const Dialect *) const = 0;

  /**
   * @brief Infer node's shape
   *
   * WARNING!!
   *
   *   Implementation SHOULD return true only when it succeeds in inference!
   *
   */
  virtual bool infer(const Node *, NodeShape &) const = 0;

  //
  // API v2
  //
  struct Context
  {
    virtual ~Context() = default;

    virtual bool known(const Node *node) const = 0;
    virtual NodeShape get(const Node *node) const = 0;
  };

  struct Sink
  {
    virtual ~Sink() = default;

    // TODO Add methods for error reporting

    // Each ShapeInferenceRule SHOULD invoke one of okay and fail before it returns
    virtual void okay(const NodeShape &) = 0;
    virtual void fail(void) = 0;
  };

  // WARNING! Invoke this method only when API v2 is supported
  virtual void infer(const Context *, const Node *, Sink *) const;
};

} // namespace loco

#endif // __LOCO_SERVICE_SHAPE_INFERENCE_RULE_H__
