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

#ifndef __TEST_HELPER_H__
#define __TEST_HELPER_H__

#include "Check.h"
#include "ProgressReporter.h"
#include "Passes.h"

#include <logo/Pass.h>
#include <logo/Phase.h>

#include <loco.h>

#include <memory>

#include <gtest/gtest.h>

/**
 * @brief Check the number of nodes in a graph starting from OUTPUTS
 */
#define EXO_TEST_ASSERT_NODE_COUNT(OUTPUTS, COUNT) \
  {                                                \
    auto v = loco::postorder_traversal(OUTPUTS);   \
    ASSERT_EQ(v.size(), (COUNT));                  \
  }

namespace exo
{
namespace test
{

/**
 * @brief Phase for test, that is used to test pass. This phase initially adds TypeInferencePass
 *        and ShapeInferencePass
 */
class TypeShapeReadyPhase
{
public:
  TypeShapeReadyPhase()
  {
    // Type and Shape inference is prerequisite for run other test
    _phase.emplace_back(std::make_unique<::exo::TypeInferencePass>());
    _phase.emplace_back(std::make_unique<::exo::ShapeInferencePass>());
  }

  template <typename PassT> void add_pass() { _phase.emplace_back(std::make_unique<PassT>()); }

  void run(loco::Graph *g)
  {
    const auto restart = logo::PhaseStrategy::Restart;
    logo::PhaseRunner<restart> phase_runner{g};

    ::exo::ProgressReporter prog(g, restart);
    phase_runner.attach(&prog);
    phase_runner.run(_phase);
  }

private:
  logo::Phase _phase;
};

/**
 * @brief Get the only succ object of type LocoNodeT. (The name `only succ` comes from English word
 *        `only child`.)
 *        parent must have 1 succ only.
 *        When there is no succ of type LocoNodeT, nullptr will be returned.
 */
template <typename LocoNodeT> inline LocoNodeT *get_only_succ(loco::Node *parent)
{
  auto succs = loco::succs(parent);
  EXO_ASSERT(succs.size() == 1, "parent has more than 1 succs.");

  return dynamic_cast<LocoNodeT *>(*succs.begin());
}

template <typename T> inline T *find_first_node_bytype(loco::Graph *g)
{
  T *first_node = nullptr;
  loco::Graph::NodeContext *nodes = g->nodes();
  uint32_t count = nodes->size();

  for (uint32_t i = 0; i < count; ++i)
  {
    first_node = dynamic_cast<T *>(nodes->at(i));
    if (first_node != nullptr)
      break;
  }

  return first_node;
}

} // namespace test
} // namespace exo

#endif // __TEST_HELPER_H__
