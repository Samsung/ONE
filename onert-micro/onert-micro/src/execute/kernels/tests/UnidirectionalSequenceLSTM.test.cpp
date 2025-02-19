/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "test_models/unidirectional_lstm/FloatUnidirectionalLSTMKernel.h"

namespace onert_micro
{
namespace execute
{
namespace testing
{

using namespace testing;

template <typename T>
std::vector<T> checkLSTMKernel(test_model::TestDataUnidirectionalLSTMBase<T> *test_data_base)
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

class UnidirectionalLSTMTest : public ::testing::Test
{
  // Do nothing
};

TEST_F(UnidirectionalLSTMTest, Float_P)
{
  onert_micro::test_model::TestDataFloatUnidirectionalLSTM test_data_kernel;
  std::vector<float> output_data_vector = checkLSTMKernel<float>(&test_data_kernel);
  EXPECT_THAT(output_data_vector,
              FloatArrayNear(test_data_kernel.get_output_data_by_index(0), 0.0000001f));
}

} // namespace testing
} // namespace execute
} // namespace onert_micro
