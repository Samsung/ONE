/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/TypeInferencePass.h"

#include <luci/IR/CircleDialect.h>
#include <luci/Service/CircleTypeInferenceRule.h>

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

namespace luci
{

bool TypeInferencePass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0; g < m->size(); ++g)
  {
    if (run(m->graph(g)))
      changed = true;
  }

  return changed;
}

bool TypeInferencePass::run(loco::Graph *g)
{
  loco::CanonicalTypeInferenceRule canonical_rule;
  luci::CircleTypeInferenceRule circle_rule;

  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
      .bind(luci::CircleDialect::get(), &circle_rule);

  return loco::apply(&rules).to(g);
}

} // namespace luci
