/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModulePhase.h"

#include "luci/Pass/CircleShapeInferencePass.h"

#include <loco.h>

#include <gtest/gtest.h>

TEST(ModulePhaseTest, saturate)
{
  auto m = luci::make_module();
  auto g = loco::make_graph();
  m->add(std::move(g));

  luci::Phase phase;

  // Any Pass will do for testing
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

  luci::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{m.get()};
  phase_runner.run(phase);

  SUCCEED();
}

TEST(ModulePhaseTest, restart)
{
  auto m = luci::make_module();
  auto g = loco::make_graph();
  m->add(std::move(g));

  luci::Phase phase;

  // Any Pass will do for testing
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

  luci::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{m.get()};
  phase_runner.run(phase);

  SUCCEED();
}
