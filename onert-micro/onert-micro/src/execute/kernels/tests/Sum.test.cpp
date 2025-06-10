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

#include "execute/OMTestUtils.h"
#include "test_models/sum/FloatSumKernel.h"
#include "test_models/sum/NegSumKernel.h"

namespace onert_micro
{
namespace execute
{
namespace testing
{

using namespace testing;

class SumTest : public ::testing::Test
{
  // Do nothing
};

template <typename T> std::vector<T> checkSumKernel(test_model::TestDataSumBase<T> *test_data_base)
{
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  interpreter.importModel(reinterpret_cast<const char *>(test_data_base->get_model_ptr()), config);

  interpreter.reset();
  interpreter.allocateInputs();

  T *input_data = reinterpret_cast<T *>(interpreter.getInputDataAt(0));

  std::copy(test_data_base->get_input_data_by_index(0).begin(),
            test_data_base->get_input_data_by_index(0).end(), input_data);
  interpreter.run(config);

  T *output_data = reinterpret_cast<T *>(interpreter.getOutputDataAt(0));
  const size_t num_elements = interpreter.getOutputSizeAt(0);
  std::vector<T> output_data_vector(output_data, output_data + num_elements);
  return output_data_vector;
}

TEST_F(SumTest, Float_P)
{
  test_model::TestDataFloatSum test_data_kernel;
  std::vector<float> output_data_vector = checkSumKernel(&test_data_kernel);
  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

TEST_F(SumTest, Input_output_type_mismatch_NEG)
{
  test_model::NegTestDataInputOutputTypeMismatchSumKernel test_data_kernel;
  EXPECT_DEATH(checkNEGSISOKernel(&test_data_kernel), "");
}

} // namespace testing
} // namespace execute
} // namespace onert_micro
