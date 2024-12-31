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
#include "test_models/reduce_prod/ReduceProdKernel.h"
#include "test_models/reduce_prod/NegReduceProdKernel.h"

namespace onert_micro
{
namespace execute
{
namespace testing
{

using namespace testing;

class ReduceProdTest : public ::testing::Test
{
  // Do nothing
};

template <typename T>
std::vector<T> checkReduceProdKernel(test_model::TestDataReduceProdBase<T> *test_data_base)
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

TEST_F(ReduceProdTest, Float_P)
{
  test_model::TestDataFloatReduceProd test_data_kernel;
  std::vector<float> output_data_vector = checkReduceProdKernel(&test_data_kernel);
  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

TEST_F(ReduceProdTest, Int_P)
{
  test_model::TestDataIntReduceProd test_data_int_reduce_prod;
  std::vector<int32_t> output_data_vector = checkReduceProdKernel(&test_data_int_reduce_prod);
  EXPECT_THAT(output_data_vector, test_data_int_reduce_prod.get_output_data_by_index(0));
}

TEST_F(ReduceProdTest, Wrong_input_type_NEG)
{
  test_model::NegTestDataWrongInputTypeReduceProdKernel test_data_kernel;
  EXPECT_DEATH(checkNEGSISOKernel(&test_data_kernel), "");
}

TEST_F(ReduceProdTest, Wrong_axis_type_NEG)
{
  test_model::NegTestDataWrongAxisTypeReduceProdKernel test_data_kernel;
  EXPECT_DEATH(checkNEGSISOKernel(&test_data_kernel), "");
}

} // namespace testing
} // namespace execute
} // namespace onert_micro
