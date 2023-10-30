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

#include <gtest/gtest.h>

#include "PatternSolver.h"

#include "core/TestHelper.h"

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

using namespace mpqsolver::pattern;

namespace
{

class CircleMPQSolverPatternSolverTest : public ::testing::Test
{
public:
  CircleMPQSolverPatternSolverTest()
  {
    char module_template[] = "CircleMPQSolverPatternSolverTest-CIRCLE-XXXXXX";
    mpqsolver::test::io_utils::makeTemporaryFile(module_template);
    _module_path = module_template;
  }

  ~CircleMPQSolverPatternSolverTest() { unlink(_module_path.c_str()); }

protected:
  mpqsolver::test::models::SoftmaxTestGraph _g;
  std::string _module_path;
};

} // namespace

TEST_F(CircleMPQSolverPatternSolverTest, verify_results)
{
  auto m = luci::make_module();
  _g.init();
  _g.transfer_to(m.get());

  // export to _module_path
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(m.get(), _module_path);
  EXPECT_TRUE(exporter.invoke(&contract));

  // create solver
  mpqsolver::pattern::PatternSolver solver(
    "uint8", "uint8",
    std::vector<QuantizationPattern>(1, QuantizationPattern::Q8SoftmaxWithQ16SubExp));

  // run solver
  auto const res = solver.run(_module_path);
  EXPECT_TRUE(res.get() != nullptr);
  ASSERT_EQ(1, res.get()->size());

  auto const graph = res.get()->graph();
  ASSERT_NE(nullptr, graph);

  uint32_t exp_count = 0;
  for (auto node : loco::postorder_traversal(loco::output_nodes(graph)))
  {
    auto const exp = dynamic_cast<luci::CircleExp *>(node);
    if (exp != nullptr)
    {
      exp_count += 1;
      auto const dtype = exp->dtype();
      // pattern was applied
      ASSERT_EQ(loco::DataType::S16, dtype);
    }
  }

  // the model has a single exp node
  ASSERT_EQ(1, exp_count);
}

TEST_F(CircleMPQSolverPatternSolverTest, empty_patterns_NEG)
{
  auto m = luci::make_module();
  _g.init();
  _g.transfer_to(m.get());

  // export to _module_path
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(m.get(), _module_path);
  EXPECT_TRUE(exporter.invoke(&contract));

  // create solver
  mpqsolver::pattern::PatternSolver solver("uint8", "uint8", std::vector<QuantizationPattern>());

  // run solver
  auto const res = solver.run(_module_path);
  EXPECT_TRUE(res.get() != nullptr);
  ASSERT_EQ(1, res.get()->size());

  auto const graph = res.get()->graph();
  ASSERT_NE(nullptr, graph);

  uint32_t exp_count = 0;
  for (auto node : loco::postorder_traversal(loco::output_nodes(graph)))
  {
    auto const exp = dynamic_cast<luci::CircleExp *>(node);
    if (exp != nullptr)
    {
      exp_count += 1;
      auto const dtype = exp->dtype();
      // pattern was not applied
      ASSERT_EQ(loco::DataType::U8, dtype);
    }
  }

  // the model has a single exp node
  ASSERT_EQ(1, exp_count);
}

TEST_F(CircleMPQSolverPatternSolverTest, empty_path_NEG)
{
  mpqsolver::pattern::PatternSolver solver(
    "uint8", "uint8",
    std::vector<QuantizationPattern>(1, QuantizationPattern::Q8LayerNormWithQ16Variance));

  EXPECT_ANY_THROW(solver.run(""));
}
