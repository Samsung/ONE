/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "execute/OMTestUtils.h"
#include "test_models/reduce_max/FloatReduceMaxKernel.h"
#include "test_models/reduce_max/NegReduceMaxKernel.h"

namespace onert_micro
{
namespace execute
{
namespace testing
{

using namespace testing;

class ReduceMaxTest : public ::testing::Test
{
  // Do nothing
};

TEST_F(ReduceMaxTest, Float_P)
{
  onert_micro::test_model::TestDataFloatReduceMax test_data_kernel;
  std::vector<float> output_data_vector =
    onert_micro::execute::testing::checkKernel<float>(1, &test_data_kernel);
  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

// TODO: enable this test
// TEST_F(ReduceMaxTest, Float_4D_2axis_P)
// {
//   onert_micro::test_model::TestDataFloatReduceMax4D_2axis test_data_kernel;
//   std::vector<float> output_data_vector =
//     onert_micro::execute::testing::checkKernel<float>(1, &test_data_kernel);
//   EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
// }

TEST_F(ReduceMaxTest, Float_3D_P)
{
  onert_micro::test_model::TestDataFloatReduceMax3D test_data_kernel;
  std::vector<float> output_data_vector =
    onert_micro::execute::testing::checkKernel<float>(1, &test_data_kernel);
  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

TEST_F(ReduceMaxTest, Mismatch_input_output_NEG)
{
  test_model::NegTestDataInputOutputTypeMismatchReduceMaxKernel test_data_kernel;
  EXPECT_DEATH(checkNEGSISOKernel(&test_data_kernel), "");
}

} // namespace testing
} // namespace execute
} // namespace onert_micro
