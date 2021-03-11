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

#ifndef __LOCO_SERVICE_SHAPE_INFERENCE_H__
#define __LOCO_SERVICE_SHAPE_INFERENCE_H__

#include "loco/Service/ShapeInferenceRule.h"
#include "loco/IR/Graph.h"

/**
 * @file This file implements dialect-agnostic shape inference framework
 *
 * HOW TO USE:
 *
 *   loco::Graph *g = ...;
 *   loco::ShapeInferenceRule *rule = ...;
 *   loco::apply(rule).to(g);
 *
 */
namespace loco
{

class ShapeInferenceSession
{
public:
  ShapeInferenceSession(const ShapeInferenceRule *rule) : _rule{rule}
  {
    // DO NOTHING
  }

public:
  bool to(Graph *g) const;

private:
  const ShapeInferenceRule *_rule;
};

inline ShapeInferenceSession apply(ShapeInferenceRule *r) { return ShapeInferenceSession{r}; }

struct ShapeInference
{
  static bool known(const Node *);
  static NodeShape get(const Node *);
  static void erase(Node *);
};

inline bool shape_known(const Node *node) { return ShapeInference::known(node); }
inline NodeShape shape_get(const Node *node) { return ShapeInference::get(node); }
inline void shape_erase(Node *node) { ShapeInference::erase(node); }

} // namespace loco

#endif // __LOCO_SERVICE_SHAPE_INFERENCE_H__
