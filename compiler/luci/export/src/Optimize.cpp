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

#include <luci/Pass/ShapeSignatureInferencePass.h>
#include <luci/Pass/CircleShapeInferencePass.h>
#include <luci/Pass/CircleTypeInferencePass.h>

// These will be removed after refactoring is finished
#include <luci/Pass/CheckCircleRulesPass.h>
#include <luci/Pass/CopyLocoItemsToCirclePass.h>
#include <luci/Pass/ShapeInferencePass.h>
#include <luci/Pass/TypeInferencePass.h>

#include <logo/Phase.h>

#include <memory>

namespace luci
{

void optimize(loco::Graph *g)
{
  logo::Phase phase;
  {
    // prepare type and shape before optimization

    // Following passes will be deprecated after refactoring is finished.
    phase.emplace_back(std::make_unique<CopyLocoItemsToCirclePass>());
    phase.emplace_back(std::make_unique<TypeInferencePass>());
    phase.emplace_back(std::make_unique<ShapeInferencePass>());

    // Following pass is for checking whether new circle rules are implemented correctly.
    // It will be deprecated after all implementation is finished.
    phase.emplace_back(std::make_unique<CheckCircleRulesPass>());

    // Following passes are needed everytime when new nodes are created.
    phase.emplace_back(std::make_unique<ShapeSignatureInferencePass>());
    phase.emplace_back(std::make_unique<CircleShapeInferencePass>());
    phase.emplace_back(std::make_unique<CircleTypeInferencePass>());

    // TODO add more optimization passes (with a knob)
  }

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace luci
