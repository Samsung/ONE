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
#include "test_models/select_v2/FloatSelectV2Kernel.h"
#include "test_models/select_v2/NegSelectV2Kernel.h"

namespace onert_micro
{
namespace
{

using namespace testing;

class SelectV2Test : public ::testing::Test
{
  // Do nothing
};

template <typename T>
std::vector<T> checkSelectV2Kernel(test_model::TestDataSelectV2Base<T> *test_data_base)
{
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  interpreter.importModel(reinterpret_cast<const char *>(test_data_base->get_model_ptr()), config);

  interpreter.reset();
  interpreter.allocateInputs();

  bool *input_cond_data = reinterpret_cast<bool *>(interpreter.getInputDataAt(0));
  T *input_x_data = reinterpret_cast<T *>(interpreter.getInputDataAt(1));
  T *input_y_data = reinterpret_cast<T *>(interpreter.getInputDataAt(2));

  std::copy(test_data_base->get_cond_input().begin(), test_data_base->get_cond_input().end(),
            input_cond_data);
  std::copy(test_data_base->get_input_data_by_index(1).begin(),
            test_data_base->get_input_data_by_index(1).end(), input_x_data);
  std::copy(test_data_base->get_input_data_by_index(2).begin(),
            test_data_base->get_input_data_by_index(2).end(), input_y_data);

  interpreter.run(config);

  T *output_data = reinterpret_cast<T *>(interpreter.getOutputDataAt(0));
  const size_t num_elements = interpreter.getOutputSizeAt(0);
  std::vector<T> output_data_vector(output_data, output_data + num_elements);
  return output_data_vector;
}

TEST_F(SelectV2Test, Float_P)
{
  test_model::TestDataFloatSelectV2 test_data_kernel;
  std::vector<float> output_data_vector = checkSelectV2Kernel(&test_data_kernel);
  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

} // namespace
} // namespace onert_micro
