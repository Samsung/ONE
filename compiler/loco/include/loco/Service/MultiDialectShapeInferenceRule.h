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

#ifndef __LOCO_SERVICE_MULTI_DIALECT_SHAPE_INFERENCE_RULE_H__
#define __LOCO_SERVICE_MULTI_DIALECT_SHAPE_INFERENCE_RULE_H__

#include "loco/Service/ShapeInferenceRule.h"

#include <map>

namespace loco
{

/**
 * @brief Shape inference rule for multiple dialects
 */
class MultiDialectShapeInferenceRule final : public ShapeInferenceRule
{
public:
  bool recognize(const Dialect *) const final;
  bool infer(const Node *, NodeShape &) const final;

  /// @brief Bind a specific rule to a Dialect
  MultiDialectShapeInferenceRule &bind(const Dialect *d, const ShapeInferenceRule *rule);

private:
  std::map<const Dialect *, const ShapeInferenceRule *> _rules;
};

} // namespace loco

#endif // __LOCO_SERVICE_MULTI_DIALECT_SHAPE_INFERENCE_RULE_H__
