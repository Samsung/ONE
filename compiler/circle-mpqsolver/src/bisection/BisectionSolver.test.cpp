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
#include "core/TestHelper.h"
#include "core/SolverOutput.h"

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

using namespace mpqsolver::test::models;

namespace
{

class CircleMPQSolverBisectionSolverTest : public ::testing::Test
{
public:
  CircleMPQSolverBisectionSolverTest()
  {
    char module_template[] = "CircleMPQSolverBisectionSolverTest-CIRCLE-XXXXXX";
    mpqsolver::test::io_utils::makeTemporaryFile(module_template);
    _module_path = module_template;

    // create data
    char hdf5_template[] = "CircleMPQSolverBisectionSolverTest-HDF5-XXXXXX";
    mpqsolver::test::hdf5_utils::createHDF5File(hdf5_template, _g._channel_size, _g._width,
                                                _g._height);
    _hdf5_path = hdf5_template;
  }

  ~CircleMPQSolverBisectionSolverTest()
  {
    unlink(_module_path.c_str());
    unlink(_hdf5_path.c_str());
  }

protected:
  AddGraph _g;
  std::string _module_path;
  std::string _hdf5_path;
};

} // namespace

TEST_F(CircleMPQSolverBisectionSolverTest, verifyResultsTest)
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
  mpqsolver::bisection::BisectionSolver solver(_hdf5_path, 0.5, "uint8", "uint8");
  solver.algorithm(mpqsolver::bisection::BisectionSolver::Algorithm::ForceQ16Back);
  SolverOutput::get().TurnOn(false);

  // run solver
  auto res = solver.run(_module_path);
  EXPECT_TRUE(res.get() != nullptr);
}

TEST_F(CircleMPQSolverBisectionSolverTest, empty_path_NEG)
{
  mpqsolver::bisection::BisectionSolver solver("", 0.0, "uint8", "uint8");
  solver.algorithm(mpqsolver::bisection::BisectionSolver::Algorithm::ForceQ16Back);
  auto const res = solver.run("");
  EXPECT_TRUE(res.get() == nullptr);
}
