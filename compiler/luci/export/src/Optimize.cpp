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

#include "Optimize.h"
#include "ProgressReporter.h"

#include <luci/Pass/ShapeInferencePass.h>
#include <luci/Pass/TypeInferencePass.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>

#include <logo/Phase.h>

#include <memory>

namespace luci
{

void optimize(loco::Graph *g)
{
  logo::Phase phase;
  {
    // prepare type and shape before optimization
    phase.emplace_back(std::make_unique<TypeInferencePass>());
    phase.emplace_back(std::make_unique<ShapeInferencePass>());
    phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
    phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

    // TODO add more optimization passes (with a knob)
  }

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace luci
