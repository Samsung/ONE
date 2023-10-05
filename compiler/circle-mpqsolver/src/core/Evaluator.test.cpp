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

#include "Evaluator.h"
#include "core/TestHelper.h"

using namespace mpqsolver::test::models;

namespace
{

class CircleMPQSolverEvaluatorTest : public ::testing::Test
{
public:
  CircleMPQSolverEvaluatorTest()
  {
    // create data
    char hdf5_template[] = "CircleMPQSolverEvaluatorTest-HDF5-XXXXXX";
    mpqsolver::test::hdf5_utils::createHDF5File(hdf5_template, _g._channel_size, _g._width,
                                                _g._height);
    _hdf5_path = hdf5_template;
  }

  ~CircleMPQSolverEvaluatorTest() { unlink(_hdf5_path.c_str()); }

protected:
  AddGraph _g;
  std::string _hdf5_path;
};

} // namespace

TEST_F(CircleMPQSolverEvaluatorTest, verifyResultsTest)
{
  // create nn module
  auto m = luci::make_module();
  _g.init();
  _g.transfer_to(m.get());

  mpqsolver::core::MAEMetric metric;
  mpqsolver::core::DatasetEvaluator evaluator(m.get(), _hdf5_path, metric);
  float value = evaluator.evaluate(m.get());
  EXPECT_FLOAT_EQ(value, 0.f);
}

TEST_F(CircleMPQSolverEvaluatorTest, empty_path_NEG)
{
  mpqsolver::core::MAEMetric metric;
  EXPECT_ANY_THROW(mpqsolver::core::DatasetEvaluator evaluator(nullptr, "", metric));
}
