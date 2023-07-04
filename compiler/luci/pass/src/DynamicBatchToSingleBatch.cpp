/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/DynamicBatchToSingleBatch.h"

#include "luci/Pass/DynamicBatchToSingleBatchPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include "ProgressReporter.h"

#include <logo/Phase.h>

namespace luci
{

void dynamic_batch_to_single_batch(luci::Module *m)
{
  assert(m); // FIX CALLER UNLESS

  for (uint32_t i = 0; i < m->size(); i++)
  {
    auto g = m->graph(i);

    logo::Phase phase;

    phase.emplace_back(std::make_unique<luci::DynamicBatchToSingleBatchPass>());

    // Needed to infer shapes of other nodes
    phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

    ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
    logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
    phase_runner.attach(&prog);
    phase_runner.run(phase);
  }
}

} // namespace luci
