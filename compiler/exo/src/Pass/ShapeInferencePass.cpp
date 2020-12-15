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

#include "ShapeInferencePass.h"

#include "Dialect/IR/TFLDialect.h"
#include "Dialect/Service/TFLShapeInferenceRule.h"

#include "Dialect/IR/CircleDialect.h"
#include "Dialect/Service/CircleShapeInferenceRule.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/CanonicalShapeInferenceRule.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/MultiDialectShapeInferenceRule.h>

#include <locoex/COpDialect.h>
#include <locoex/Service/COpShapeInferenceRule.h>

namespace exo
{

/**
 * @note  Currently, TFL and Circle backend share this inference. However, TFL
 *        backend does not require rule for Circle dialect.
 *        TODO Make dedicated inference pass for Circle Dialect.
 */
bool ShapeInferencePass::run(loco::Graph *g)
{
  loco::CanonicalShapeInferenceRule canonical_rule;
  locoex::TFLShapeInferenceRule tfl_rule;
  locoex::CircleShapeInferenceRule circle_rule;
  locoex::COpShapeInferenceRule cop_rule;

  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(locoex::TFLDialect::get(), &tfl_rule)
    .bind(locoex::CircleDialect::get(), &circle_rule)
    .bind(locoex::COpDialect::get(), &cop_rule);

  return loco::apply(&rules).to(g);
}

} // namespace exo
