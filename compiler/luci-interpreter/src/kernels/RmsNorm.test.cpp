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
#include "kernels/RmsNorm.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class RmsNormTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(RmsNormTest, Simple)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 2, 1}, {0, 1, 2, 3}, _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  RmsNormParams params{};
  params.epsilon = 0.00001f;

  RmsNorm kernel(&input_tensor, &gamma_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear({0, 1, 1, 1}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 1}));
}

TEST_F(RmsNormTest, Default_gamma)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7},
                                                           _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  RmsNormParams params{};
  params.epsilon = 0.001f;

  RmsNorm kernel(&input_tensor, &gamma_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(
    extractTensorData<float>(output_tensor),
    FloatArrayNear({0, 1.412802, 0.784404, 1.176606, 0.883431, 1.104288, 0.920347, 1.073738}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 2}));
}

TEST_F(RmsNormTest, Have_gamma)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7},
                                                           _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  RmsNormParams params{};
  params.epsilon = 0.001f;

  RmsNorm kernel(&input_tensor, &gamma_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(
    extractTensorData<float>(output_tensor),
    FloatArrayNear({0, 2.825603, 1.568808, 2.353213, 1.766861, 2.208577, 1.840694, 2.147477}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 2}));
}

TEST_F(RmsNormTest, Wrong_gamma_dim_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7},
                                                           _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({3}, {1, 1, 1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  RmsNormParams params{};
  params.epsilon = 0.001f;

  RmsNorm kernel(&input_tensor, &gamma_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(RmsNormTest, Unsupported_dims_NEG)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 2}, {0, 1, 2, 3}, _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  RmsNormParams params{};
  params.epsilon = 0.001f;

  RmsNorm kernel(&input_tensor, &gamma_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
