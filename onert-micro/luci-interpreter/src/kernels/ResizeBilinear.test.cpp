/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved
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
#include "loader/ModuleLoader.h"
#include "luci_interpreter/test_models/resize_bilinear/FloatResizeBilinearKernel.h"
#include "luci_interpreter/test_models/resize_bilinear/U8ResizeBilinearKernel.h"
#include "luci_interpreter/test_models/resize_bilinear/NegResizeBilinearKernel.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ResizeBilinearTest : public ::testing::Test
{
  // Do nothing
};

template <typename T>
std::vector<T> checkResizeBilinearKernel(test_kernel::TestDataBase<T> *test_data_base)
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

TEST_F(ResizeBilinearTest, Float_P)
{
  test_kernel::TestDataFloatResizeBilinear test_data_kernel(false);
  std::vector<float> output_data_vector = checkResizeBilinearKernel(&test_data_kernel);

  EXPECT_THAT(output_data_vector,
              FloatArrayNear(test_data_kernel.get_output_data_by_index(0), 0.0001f));
}

TEST_F(ResizeBilinearTest, HalfPixelCenter_Float_P)
{

  test_kernel::TestDataFloatResizeBilinear test_data_kernel(true);
  std::vector<float> output_data_vector = checkResizeBilinearKernel(&test_data_kernel);

  EXPECT_THAT(output_data_vector,
              FloatArrayNear(test_data_kernel.get_output_data_by_index(0), 0.0001f));
}

TEST_F(ResizeBilinearTest, Uint8_P)
{
  test_kernel::TestDataUint8ResizeBilinear test_data_kernel(false);
  std::vector<uint8_t> output_data_vector = checkResizeBilinearKernel<uint8_t>(&test_data_kernel);

  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

TEST_F(ResizeBilinearTest, HalfPixelCenter_Uint8_P)
{
  test_kernel::TestDataUint8ResizeBilinear test_data_kernel(true);
  std::vector<uint8_t> output_data_vector = checkResizeBilinearKernel<uint8_t>(&test_data_kernel);

  EXPECT_THAT(output_data_vector, test_data_kernel.get_output_data_by_index(0));
}

TEST_F(ResizeBilinearTest, InvalidInputShape_Float_NEG)
{

  test_kernel::NegTestDataInvalidInputShapeFloatResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(ResizeBilinearTest, InvalidParams_Float_NEG)
{

  test_kernel::NegTestDataInvalidParamFloatResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(ResizeBilinearTest, InvalidSizeShape_Float_NEG)
{

  test_kernel::NegTestDataInvalidSizeShapeDimensionsFloatResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(ResizeBilinearTest, InvalidInputShape_uint8_NEG)
{

  test_kernel::NegTestDataInvalidInputShapeUint8ResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(ResizeBilinearTest, InvalidParams_uint8_NEG)
{

  test_kernel::NegTestDataInvalidParamUint8ResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

TEST_F(ResizeBilinearTest, InvalidSizeShape_uint8_NEG)
{

  test_kernel::NegTestDataInvalidSizeShapeDimensionsUint8ResizeBilinearKernel test_data_kernel;

  MemoryManager memory_manager{};
  RuntimeModule runtime_module{};
  bool dealloc_input = true;
  // Load model with single op
  auto *model_data_raw = reinterpret_cast<const char *>(test_data_kernel.get_model_ptr());
  EXPECT_DEATH(ModuleLoader::load(&runtime_module, &memory_manager, model_data_raw, dealloc_input),
               "");
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
