/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/TestUtils.h"
#include "luci_interpreter/test_models/split_v/SplitVKernel.h"

#include "loader/ModuleLoader.h"

namespace luci_interpreter
{
namespace
{

using namespace testing;

class SplitVTest : public ::testing::Test
{
  // Do nothing
};

template <typename T>
std::vector<std::vector<T>> checkSplitKernel(test_kernel::TestDataBase<T> *test_data_base)
{
  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;

  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_base->get_model_ptr());
  ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input);

  auto *main_runtime_graph = runtime_module.getMainGraph();
  assert(main_runtime_graph->getNumOfInputTensors() == 1);

  // Set input data
  {
    auto *input_tensor_data = reinterpret_cast<T *>(main_runtime_graph->configureGraphInput(0));
    std::copy(test_data_base->get_input_data_by_index(0).begin(),
              test_data_base->get_input_data_by_index(0).end(), input_tensor_data);
  }

  runtime_module.execute();

  assert(main_runtime_graph->getNumOfOutputTensors() == 3);

  std::vector<std::vector<T>> result;

  for (int i = 0; i < 3; ++i)
  {
    T *output_data = reinterpret_cast<T *>(main_runtime_graph->getOutputDataByIndex(i));
    const size_t num_elements = (main_runtime_graph->getOutputDataSizeByIndex(i) / sizeof(T));
    std::vector<T> output_data_vector(output_data, output_data + num_elements);
    result.push_back(output_data_vector);
  }

  return result;
}

TEST_F(SplitVTest, MainTest_P)
{
  test_kernel::TestDataSplitVKernel<float> test_data_kernel;
  const auto output_data_vector = checkSplitKernel(&test_data_kernel);

  for (int i = 0; i < 3; ++i)
  {
    EXPECT_THAT(output_data_vector[i], test_data_kernel.get_output_data_by_index(i));
  }
}

// TODO: add negative tests?

} // namespace
} // namespace luci_interpreter
