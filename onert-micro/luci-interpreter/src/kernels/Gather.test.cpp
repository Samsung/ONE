/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci_interpreter/test_models/gather/FloatGatherKernel.h"
#include "luci_interpreter/test_models/gather/IntGatherKernel.h"

#include "loader/ModuleLoader.h"

namespace luci_interpreter
{
namespace
{

using namespace testing;

class GatherTest : public ::testing::Test
{
  // Do nothing
};

template <typename T> std::vector<T> checkGatherKernel(test_kernel::TestDataBase<T> *test_data_base)
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

  assert(main_runtime_graph->getNumOfOutputTensors() == 1);

  T *output_data = reinterpret_cast<T *>(main_runtime_graph->getOutputDataByIndex(0));
  const size_t num_elements = (main_runtime_graph->getOutputDataSizeByIndex(0) / sizeof(T));
  std::vector<T> output_data_vector(output_data, output_data + num_elements);
  return output_data_vector;
}

TEST_F(GatherTest, Gather_Float_P)
{
  test_kernel::TestDataFloatGather test_data_float_gather;
  std::vector<float> output_data_vector = checkGatherKernel(&test_data_float_gather);
  EXPECT_THAT(output_data_vector, kernels::testing::FloatArrayNear(
                                    test_data_float_gather.get_output_data_by_index(0), 0.0001f));
}

TEST_F(GatherTest, Gather_Int_P)
{
  test_kernel::TestDataIntGather test_data_int_gather;
  std::vector<int32_t> output_data_vector = checkGatherKernel(&test_data_int_gather);
  EXPECT_THAT(output_data_vector, test_data_int_gather.get_output_data_by_index(0));
}

// TODO: add negative tests?
// TODO: add S8 test

} // namespace
} // namespace luci_interpreter
