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

#include "Optimizer.h"

#include "Knob.h"
#include "ProgressReporter.h"
#include "Transforms.h"

#include <logo/Phase.h>

#include <memory>

namespace moco
{
namespace tf
{

void Optimizer::optimize(loco::Graph *g) const
{
  logo::Phase phase;

  /* TRANSFORM DECLARATION BEGIN */
  // Shape inference is required for ResolveRedundantReshape
  phase.emplace_back(std::make_unique<ShapeInferencePass>());

  if (moco::tf::get<moco::tf::Knob::ConstantFolding>())
  {
    phase.emplace_back(std::make_unique<logo::ConstantFoldingPass>());
  }

  if (moco::tf::get<moco::tf::Knob::RemoveDeadNode>())
  {
    phase.emplace_back(std::make_unique<logo::RemoveDeadNodePass>());
  }

  if (moco::tf::get<moco::tf::Knob::ReorderDecode>() &&
      moco::tf::get<moco::tf::Knob::ReorderDecodeTensorBiasAdd>())
  {
    phase.emplace_back(std::make_unique<logo::ReorderDecodePass<loco::TensorBiasAdd>>());
  }

  if (moco::tf::get<moco::tf::Knob::ReorderDecode>() &&
      moco::tf::get<moco::tf::Knob::ReorderDecodeReLU>())
  {
    phase.emplace_back(std::make_unique<logo::ReorderDecodePass<loco::ReLU>>());
  }

  if (moco::tf::get<moco::tf::Knob::SimplifyDomainConversion>())
  {
    phase.emplace_back(std::make_unique<logo::SimplifyDomainConversionPass>());
  }

  if (moco::tf::get<moco::tf::Knob::RemoveForwardNode>())
  {
    phase.emplace_back(std::make_unique<logo::RemoveForwardNodePass>());
  }

  if (moco::tf::get<moco::tf::Knob::ResolveDuplicateReshape>())
  {
    phase.emplace_back(std::make_unique<logo::ResolveDuplicateReshapePass>());
  }

  if (moco::tf::get<moco::tf::Knob::ResolveRedundantReshape>())
  {
    phase.emplace_back(std::make_unique<logo::ResolveRedundantReshapePass>());
  }
  /* TRANSFORM DECLARATION END */

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace tf
} // namespace moco
