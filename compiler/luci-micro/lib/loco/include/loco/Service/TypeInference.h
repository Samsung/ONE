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

#ifndef __LOCO_SERVICE_TYPE_INFERENCE_H__
#define __LOCO_SERVICE_TYPE_INFERENCE_H__

#include "loco/IR/DataType.h"

#include "loco/IR/Node.h"
#include "loco/IR/Dialect.h"
#include "loco/IR/Graph.h"

#include <map>

/**
 * @file This file implements dialect-agnostic type inference framework.
 *
 * HOW TO USE:
 *
 *   loco::Graph *g = ...;
 *   loco::TypeInferenceRule *rule = ...;
 *   loco::apply(rule).to(g);
 *
 */
namespace loco
{

struct TypeInferenceRule
{
  virtual ~TypeInferenceRule() = default;

  /// @brief Return true if this rule recognizes a given dialect
  virtual bool recognize(const Dialect *) const = 0;

  /**
   * Framework guarantees the followings:
   *
   * 1. Framework tries to infer the data type of each node only after the data type of all of
   *    its valid (= non-nullptr) argument nodes is inferred.
   * 2. The result of preceding "infer" is accessible through below dtype_get method.
   *    - This holds only when preceding "infer" returns true.
   */
  virtual bool infer(const Node *, DataType &) const = 0;
};

/**
 * @brief Type Inference Rule for Canonical Dialect
 */
struct CanonicalTypeInferenceRule final : public TypeInferenceRule
{
  bool recognize(const Dialect *) const final;
  bool infer(const Node *, DataType &) const final;
};

/**
 * @brief Type Inference Rule for multiple dialects
 */
class MultiDialectTypeInferenceRule final : public TypeInferenceRule
{
public:
  bool recognize(const Dialect *) const final;
  bool infer(const Node *, DataType &) const final;

  /// @brief Bind a specific rule to a Dialect
  MultiDialectTypeInferenceRule &bind(const Dialect *d, const TypeInferenceRule *rule);

private:
  std::map<const Dialect *, const TypeInferenceRule *> _rules;
};

class TypeInferenceSession
{
public:
  TypeInferenceSession(const TypeInferenceRule *rule) : _rule{rule}
  {
    // DO NOTHING
  }

public:
  bool to(Graph *g) const;

private:
  const TypeInferenceRule *_rule;
};

inline TypeInferenceSession apply(TypeInferenceRule *r) { return TypeInferenceSession{r}; }

struct TypeInference
{
  static bool known(const Node *);
  static DataType get(const Node *);
  static void erase(Node *);
};

inline bool dtype_known(const Node *node) { return TypeInference::known(node); }
inline DataType dtype_get(const Node *node) { return TypeInference::get(node); }
inline void dtype_erase(Node *node) { TypeInference::erase(node); }

} // namespace loco

#endif // __LOCO_SERVICE_TYPE_INFERENCE_H__
