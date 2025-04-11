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

#include "test_models/logical_not/BoolLogicalNotKernel.h"
#include "test_models/logical_not/NegLogicalNotKernel.h"

namespace onert_micro::execute::testing
{

// ------------------------------------------------------------------------------------------------

class LogicalNotTest : public ::testing::Test
{
};

// ------------------------------------------------------------------------------------------------

TEST_F(LogicalNotTest, Bool_P)
{
  test_model::TestDataBoolLogicalNot test_data_bool_logical_not;
  auto output_data_vector = checkKernel<bool>(1, &test_data_bool_logical_not);

  EXPECT_TRUE(output_data_vector == test_data_bool_logical_not.get_output_data_by_index(0));
}

TEST_F(LogicalNotTest, OutputTypeMismatch_NEG)
{
  test_model::NegTestDataLogicalNotOutputTypeMismatch neg_test_data;

  EXPECT_DEATH(checkNEGSISOKernel(&neg_test_data), "");
}

TEST_F(LogicalNotTest, OutputShapeMismatch_NEG)
{
  test_model::NegTestDataLogicalNotOutputShapeMismatch neg_test_data;

  EXPECT_DEATH(checkNEGSISOKernel(&neg_test_data), "");
}

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::execute::testing
