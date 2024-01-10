/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#include "luci_interpreter/test_models/sub/FloatSubKernel.h"
#include "luci_interpreter/test_models/sub/IntSubKernel.h"
#include "luci_interpreter/test_models/sub/NegSubKernel.h"

#include "loader/ModuleLoader.h"

namespace luci_interpreter
{
namespace
{

using namespace testing;

class SubTest : public ::testing::Test
{
  // Do nothing
};

template <typename T> std::vector<T> checkSubKernel(test_kernel::TestDataBase<T> *test_data_base)
{
  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;

  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_base->get_model_ptr());
  ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input);

  auto *main_runtime_graph = runtime_module.getMainGraph();
  assert(main_runtime_graph->getNumOfInputTensors() == 2);

  // set left input data
  {
    auto *input_tensor_data = reinterpret_cast<T *>(main_runtime_graph->configureGraphInput(0));
    std::copy(test_data_base->get_input_data_by_index(0).begin(),
              test_data_base->get_input_data_by_index(0).end(), input_tensor_data);
  }

  // set right input data
  {
    auto *input_tensor_data = reinterpret_cast<T *>(main_runtime_graph->configureGraphInput(1));
    std::copy(test_data_base->get_input_data_by_index(1).begin(),
              test_data_base->get_input_data_by_index(1).end(), input_tensor_data);
  }

  runtime_module.execute();

  assert(main_runtime_graph->getNumOfOutputTensors() == 1);

  T *output_data = reinterpret_cast<T *>(main_runtime_graph->getOutputDataByIndex(0));
  const size_t num_elements = (main_runtime_graph->getOutputDataSizeByIndex(0) / sizeof(T));
  std::vector<T> output_data_vector(output_data, output_data + num_elements);
  return output_data_vector;
}

TEST_F(SubTest, Float_P)
{
  // No broadcast
  {
    const bool is_with_broadcast = false;
    test_kernel::TestDataFloatSub test_data_float_kernel(is_with_broadcast);
    std::vector<float> output_data_vector = checkSubKernel(&test_data_float_kernel);
    EXPECT_THAT(output_data_vector, kernels::testing::FloatArrayNear(
                                      test_data_float_kernel.get_output_data_by_index(0), 0.0001f));
  }
  // With broadcast
  {
    const bool is_with_broadcast = true;
    test_kernel::TestDataFloatSub test_data_float_kernel(is_with_broadcast);
    std::vector<float> output_data_vector = checkSubKernel(&test_data_float_kernel);
    EXPECT_THAT(output_data_vector, kernels::testing::FloatArrayNear(
                                      test_data_float_kernel.get_output_data_by_index(0), 0.0001f));
  }
}

TEST_F(SubTest, INT_P)
{
  // No broadcast
  {
    const bool is_with_broadcast = false;
    test_kernel::TestDataIntSub test_data_kernel(is_with_broadcast);
    const auto output_data_vector = checkSubKernel<int32_t>(&test_data_kernel);
    EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
  }
  // With broadcast
  {
    const bool is_with_broadcast = true;
    test_kernel::TestDataIntSub test_data_kernel(is_with_broadcast);
    const auto output_data_vector = checkSubKernel<int32_t>(&test_data_kernel);
    EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
  }
}

TEST_F(SubTest, Inputs_type_mismatch_NEG)
{
  test_kernel::NegTestDataInputsTypeMismatchSubKernel test_data_kernel;
  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(SubTest, Input_output_type_mismatch_NEG)
{
  test_kernel::NegTestDataInputOutputTypeMismatchSubKernel test_data_kernel;
  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(SubTest, No_quant_params_NEG)
{
  test_kernel::NegTestDataNoQuantParamsSubKernel test_data_kernel;
  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

// TODO: add tests for U8 and S16
// TODO: add tests for inplace optimizations for all types

} // namespace
} // namespace luci_interpreter

#include "PALSub.h"

#include "BinaryOpCommon.h"

#include <array>
#include <numeric>

namespace luci_interpreter
{
namespace
{

class PALSubTest : public ::testing::Test
{
  // Do nothing
};

TEST_F(PALSubTest, Float_P)
{
  // No broadcast
  {
    const bool is_with_broadcast = false;
    test_kernel::TestDataFloatSub test_data_kernel(is_with_broadcast);

    luci_interpreter_pal::ArithmeticParams params{};
    kernels::fillArithmeticActivationRange<float>(params, kernels::Activation::NONE);

    const auto &input1 = test_data_kernel.get_input_data_by_index(0);
    const auto &input2 = test_data_kernel.get_input_data_by_index(1);

    const auto num_elements = input1.size();
    EXPECT_EQ(num_elements, input2.size());

    std::vector<float> output = std::vector<float>(num_elements);
    luci_interpreter_pal::Sub<float>(params, num_elements, input1.data(), input2.data(),
                                     const_cast<float *>(output.data()));

    EXPECT_THAT(output, kernels::testing::FloatArrayNear(
                          test_data_kernel.get_output_data_by_index(0), 0.0001f));
  }

  // With broadcast
  {
    const bool is_with_broadcast = true;
    test_kernel::TestDataFloatSub test_data_kernel(is_with_broadcast);

    luci_interpreter_pal::ArithmeticParams params{};
    kernels::fillArithmeticActivationRange<float>(params, kernels::Activation::NONE);

    const auto &input1 = test_data_kernel.get_input_data_by_index(0);
    const auto &input2 = test_data_kernel.get_input_data_by_index(1);

    const int32_t shape[2] = {2, 5};
    const int32_t shape_broadcast[2] = {2, 1};

    assert(input1.size() ==
           std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<float>()));
    assert(input2.size() == std::accumulate(std::begin(shape_broadcast), std::end(shape_broadcast),
                                            1, std::multiplies<float>()));

    std::vector<float> output = std::vector<float>(
      std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<float>()));
    luci_interpreter_pal::BroadcastSub4DSlow<float>(
      params, RuntimeShape{2, shape}, input1.data(), RuntimeShape{2, shape_broadcast},
      input2.data(), RuntimeShape{2, shape}, const_cast<float *>(output.data()));

    EXPECT_THAT(output, kernels::testing::FloatArrayNear(
                          test_data_kernel.get_output_data_by_index(0), 0.0001f));
  }
}

} // namespace
} // namespace luci_interpreter
