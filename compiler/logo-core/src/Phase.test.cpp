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

#include <logo/Phase.h>

#include <loco.h>

#include <gtest/gtest.h>

namespace
{

struct Bumblebee final : public logo::Pass
{
  const char *name(void) const final { return "Bee"; }
  bool run(loco::Graph *) final { return false; }
};

} // namespace

TEST(LogoPhaseSaturateTests, simple)
{
  loco::Graph g;
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{&g};
  logo::Phase phase;

  phase.emplace_back(std::make_unique<Bumblebee>());
  phase_runner.run(phase);

  SUCCEED();
}

TEST(LogoPhaseRestartTests, simple)
{
  loco::Graph g;
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{&g};
  logo::Phase phase;

  phase.emplace_back(std::make_unique<Bumblebee>());
  phase_runner.run(phase);

  SUCCEED();
}
