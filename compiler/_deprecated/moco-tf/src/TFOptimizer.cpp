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

#include "TFOptimizer.h"

#include "Knob.h"
#include "ProgressReporter.h"
#include "Transforms.h"

#include <logo/Phase.h>

#include <memory>

namespace moco
{
namespace tf
{

void TFOptimizer::optimize(loco::Graph *g) const
{
  logo::Phase phase;

  /* TRANSFORM DECLARATION BEGIN */
  if (moco::tf::get<moco::tf::Knob::ResolveFusedBatchNorm>())
  {
    phase.emplace_back(std::make_unique<moco::ResolveFusedBatchNorm>());
  }
  if (moco::tf::get<moco::tf::Knob::FuseBinaryIntoPreceding>())
  {
    phase.emplace_back(std::make_unique<moco::FuseBinaryIntoPreceding>());
  }
  if (moco::tf::get<moco::tf::Knob::ResolveConstantShape>())
  {
    phase.emplace_back(std::make_unique<moco::ResolveConstantShape>());
  }
  if (moco::tf::get<moco::tf::Knob::ResolveReshapeWildcardDim>())
  {
    phase.emplace_back(std::make_unique<moco::ResolveReshapeWildcardDim>());
  }
  if (moco::tf::get<moco::tf::Knob::ResolveSquaredDifference>())
  {
    phase.emplace_back(std::make_unique<moco::ResolveSquaredDifference>());
  }
  if (moco::tf::get<moco::tf::Knob::RemoveTFIdentityNode>())
  {
    phase.emplace_back(std::make_unique<moco::RemoveTFIdentityNode>());
  }
  if (moco::tf::get<moco::tf::Knob::RemoveDeadNode>())
  {
    phase.emplace_back(std::make_unique<logo::RemoveDeadNodePass>());
  }
  if (moco::tf::get<moco::tf::Knob::SqueezeReduceNode>())
  {
    phase.emplace_back(std::make_unique<moco::SqueezeReduceNode>());
  }
  // Shape inference is needed for added nodes doing above transformations
  phase.emplace_back(std::make_unique<moco::tf::ShapeInferencePass>());
  phase.emplace_back(std::make_unique<moco::tf::TypeInferencePass>());
  /* TRANSFORM DECLARATION END */

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace tf
} // namespace moco
