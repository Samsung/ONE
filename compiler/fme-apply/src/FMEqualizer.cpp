/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FMEqualizer.h"
#include "InsertScaleShift.h"
#include "EqualizePatternCheck.h"
#include "pass/FusePostScalePass.h"
#include "pass/FusePreScalePass.h"
#include "ProgressReporter.h"

#include <luci/IR/CircleNode.h>

#include <logo/Phase.h>
#include <logo/RemoveDeadNodeWithQueryPass.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>

using namespace fme_apply;

#define THROW_UNLESS(COND, MSG) \
  if (not(COND))                \
    throw std::runtime_error(MSG);

namespace
{

// Throw exception if virtual Op (Scale) exists
void check_no_scale(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    const auto custom = dynamic_cast<const luci::CircleCustom *>(node);
    if (custom == nullptr)
      continue;

    if (custom->numInputs() != 2)
      continue;

    const auto code = custom->custom_code();
    if (code == "scale")
      throw std::runtime_error("Virtual node(" + code + ") remains.");
  }
}

} // namespace

namespace fme_apply
{

void FMEqualizer::equalize(loco::Graph *g, std::vector<EqualizePattern> &p)
{
  THROW_UNLESS(g != nullptr, "Invalid argument g");

  // Check the patterns are valid on the graph
  check_patterns_valid(g, p);

  // Insert Scale/Shift based on patterns
  InsertScaleShift issp(p);
  issp.run(g);

  logo::Phase phase;

  // Default passes
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  // Fuse Pre/Post Scale
  phase.emplace_back(std::make_unique<fme_apply::FusePreScalePass>());
  phase.emplace_back(std::make_unique<fme_apply::FusePostScalePass>());

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);

  // Check if all Scale/Shift nodes are removed
  check_no_scale(g);
}

} // namespace fme_apply

#undef THROW_UNLESS
