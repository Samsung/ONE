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

#include "TypeInferencePass.h"

#include "Dialect/IR/TFLDialect.h"
#include "Dialect/Service/TFLTypeInferenceRule.h"

#include "Dialect/IR/CircleDialect.h"
#include "Dialect/Service/CircleTypeInferenceRule.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <locoex/COpDialect.h>
#include <locoex/Service/COpTypeInference.h>

namespace exo
{

/**
 * @note  Currently, TFL and Circle backend share this inference. However, TFL
 *        backend does not require rule for Circle dialect.
 *        TODO Make dedicated inference pass for Circle Dialect.
 */
bool TypeInferencePass::run(loco::Graph *g)
{
  loco::CanonicalTypeInferenceRule canonical_rule;
  locoex::TFLTypeInferenceRule tfl_rule;
  locoex::CircleTypeInferenceRule circle_rule;
  locoex::COpTypeInferenceRule cop_rule;

  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(locoex::TFLDialect::get(), &tfl_rule)
    .bind(locoex::CircleDialect::get(), &circle_rule)
    .bind(locoex::COpDialect::get(), &cop_rule);

  return loco::apply(&rules).to(g);
}

} // namespace exo
