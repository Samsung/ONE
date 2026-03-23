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

#include <moco/IR/TFDialect.h>

#include <moco/Service/TFTypeInferenceRule.h>

#include <loco.h>

#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <locoex/COpDialect.h>
#include <locoex/Service/COpTypeInference.h>

namespace moco
{
namespace tf
{

bool TypeInferencePass::run(loco::Graph *graph)
{
  loco::CanonicalTypeInferenceRule canonical_rule;
  moco::TFTypeInferenceRule tf_rule;     // rule for TF dialect
  locoex::COpTypeInferenceRule cop_rule; // rule for custop op

  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(TFDialect::get(), &tf_rule)
    .bind(locoex::COpDialect::get(), &cop_rule);

  loco::apply(&rules).to(graph);

  return false;
}

} // namespace tf
} // namespace moco
