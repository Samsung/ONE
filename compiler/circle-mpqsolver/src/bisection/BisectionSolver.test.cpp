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

#include "BisectionSolver.h"

#include "core/SolverOutput.h"
#include "core/TestHelper.h"

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

namespace
{

class CircleMPQSolverBisectionSolverTestF : public ::testing::Test
{
public:
  CircleMPQSolverBisectionSolverTestF()
  {
    char module_template[] = "CircleMPQSolverBisectionSolverTest-CIRCLE-XXXXXX";
    mpqsolver::test::io_utils::makeTemporaryFile(module_template);
    _module_path = module_template;
  }

  ~CircleMPQSolverBisectionSolverTestF() { unlink(_module_path.c_str()); }

protected:
  mpqsolver::test::models::AddGraph _g;
  std::string _module_path;
};

} // namespace

TEST_F(CircleMPQSolverBisectionSolverTestF, verifyResultsTest)
{
  // create network
  auto m = luci::make_module();
  _g.init();
  _g.transfer_to(m.get());

  // export to _module_path
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(m.get(), _module_path);
  EXPECT_TRUE(exporter.invoke(&contract));

  // create solver
  mpqsolver::core::Quantizer::Context ctx;
  mpqsolver::bisection::BisectionSolver solver(ctx, 0.5);
  auto data = mpqsolver::test::data_utils::getSingleDataProvider();
  solver.setInputData(std::move(data));
  solver.algorithm(mpqsolver::bisection::BisectionSolver::Algorithm::ForceQ16Back);
  SolverOutput::get().TurnOn(false);

  // run solver
  auto res = solver.run(_module_path);
  EXPECT_TRUE(res.get() != nullptr);
}

TEST(CircleMPQSolverBisectionSolverTest, empty_path_NEG)
{
  mpqsolver::core::Quantizer::Context ctx;
  mpqsolver::bisection::BisectionSolver solver(ctx, 0.0);
  solver.algorithm(mpqsolver::bisection::BisectionSolver::Algorithm::ForceQ16Back);
  EXPECT_ANY_THROW(solver.run(""));
}
