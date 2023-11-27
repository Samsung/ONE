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

#include "DataProvider.h"
#include "TestHelper.h"

TEST(CircleMPQSolverEvaluatorTest, verifyResultsTest)
{
  // create nn module
  auto m = luci::make_module();
  mpqsolver::test::models::AddGraph g;
  g.init();
  g.transfer_to(m.get());

  mpqsolver::core::MAEMetric metric;
  auto data = mpqsolver::test::data_utils::getSingleDataProvider();
  mpqsolver::core::DatasetEvaluator evaluator(m.get(), *data.get(), metric);
  float value = evaluator.evaluate(m.get());
  EXPECT_FLOAT_EQ(value, 0.f);
}

TEST(CircleMPQSolverEvaluatorTest, empty_path_NEG)
{
  mpqsolver::core::MAEMetric metric;
  EXPECT_ANY_THROW(mpqsolver::core::H5FileDataProvider data("", "");
                   mpqsolver::core::DatasetEvaluator evaluator(nullptr, data, metric));
}
