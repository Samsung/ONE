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

#include "test_models/cum_sum/FloatCumSumKernel.h"
#include "test_models/cum_sum/IntCumSumKernel.h"
#include "test_models/cum_sum/NegCumSumKernel.h"

namespace onert_micro::execute::testing
{

// ------------------------------------------------------------------------------------------------

class CumSumTest : public ::testing::Test
{
};

// ------------------------------------------------------------------------------------------------

template <typename T>
std::vector<T> checkCumSumKernel(test_model::TestDataCumSumBase<T> *test_data_base)
{
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  auto model_ptr = reinterpret_cast<const char *>(test_data_base->get_model_ptr());

  interpreter.importModel(model_ptr, config);
  interpreter.reset();
  interpreter.allocateInputs();

  T *input_data = reinterpret_cast<T *>(interpreter.getInputDataAt(0));
  const auto &input_data_ref = test_data_base->get_input_data_by_index(0);

  std::copy(input_data_ref.begin(), input_data_ref.end(), input_data);

  auto num_inputs = interpreter.getNumberOfInputs();

  if (num_inputs >= 2)
  {
    uint8_t *axis_data = reinterpret_cast<uint8_t *>(interpreter.getInputDataAt(1));

    int32_t size;
    const uint8_t *axis_data_ref = test_data_base->get_input_data_ptr_by_index(1, size);

    if (size > 0)
    {
      std::copy(axis_data_ref, axis_data_ref + size, axis_data);
    }
  }

  interpreter.run(config);

  T *output_data = reinterpret_cast<T *>(interpreter.getOutputDataAt(0));
  const size_t num_elements = interpreter.getOutputSizeAt(0);

  std::vector<T> output_data_vector(output_data, output_data + num_elements);

  return output_data_vector;
}

// ------------------------------------------------------------------------------------------------

TEST_F(CumSumTest, Float)
{
  test_model::TestDataFloatCumSum test_data;
  auto output_data_vector = checkCumSumKernel<float>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, Int32)
{
  test_model::TestDataInt32CumSum test_data;
  auto output_data_vector = checkCumSumKernel<int32_t>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, Int64)
{
  test_model::TestDataInt64CumSum test_data;
  auto output_data_vector = checkCumSumKernel<int64_t>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, Float_TwoInputs)
{
  test_model::TestDataTwoInputsFloatCumSum test_data;
  auto output_data_vector = checkCumSumKernel<float>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, Float_Reverse)
{
  test_model::TestDataReverseFloatCumSum test_data;
  auto output_data_vector = checkCumSumKernel<float>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, Float_Exclusive)
{
  test_model::TestDataExclusiveFloatCumSum test_data;
  auto output_data_vector = checkCumSumKernel<float>(&test_data);

  EXPECT_TRUE(output_data_vector == test_data.get_output_data_by_index(0));
}

TEST_F(CumSumTest, AxisShapeMismatch_NEG)
{
  test_model::NegTestDataCumSumAxisShapeMismatch neg_test_data;

  EXPECT_DEATH(checkNEGSISOKernel(&neg_test_data), "");
}

TEST_F(CumSumTest, AxisTypeMismatch_NEG)
{
  test_model::NegTestDataCumSumAxisTypeMismatch neg_test_data;

  EXPECT_DEATH(checkNEGSISOKernel(&neg_test_data), "");
}

TEST_F(CumSumTest, InputShapeMismatch_NEG)
{
  test_model::NegTestDataCumSumInputShapeMismatch neg_test_data;

  EXPECT_DEATH(checkNEGSISOKernel(&neg_test_data), "");
}

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::execute::testing
