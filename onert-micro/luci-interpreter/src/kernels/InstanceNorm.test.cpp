/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#include "kernels/InstanceNorm.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class InstanceNormTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(InstanceNormTest, Simple)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 2, 1}, {1, 1, 1, 1}, _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor beta_tensor = makeInputTensor<DataType::FLOAT32>({1}, {2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  InstanceNormParams params{};
  params.epsilon = 0.1f;
  params.activation = Activation::NONE;

  InstanceNorm kernel(&input_tensor, &gamma_tensor, &beta_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear({2, 2, 2, 2}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 1}));
}

TEST_F(InstanceNormTest, Single_gamma_beta)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 1, 2}, {1, 1, 1, 1}, _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor beta_tensor = makeInputTensor<DataType::FLOAT32>({1}, {2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  InstanceNormParams params{};
  params.epsilon = 0.1f;
  params.activation = Activation::NONE;

  InstanceNorm kernel(&input_tensor, &gamma_tensor, &beta_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear({2, 2, 2, 2}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 1, 2}));
}

TEST_F(InstanceNormTest, Wrong_gamma_beta_dim_NEG)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 1, 2}, {1, 1, 1, 1}, _memory_manager.get());
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({3}, {1, 1, 1}, _memory_manager.get());
  Tensor beta_tensor = makeInputTensor<DataType::FLOAT32>({3}, {2, 2, 2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  InstanceNormParams params{};
  params.epsilon = 0.1f;
  params.activation = Activation::NONE;

  InstanceNorm kernel(&input_tensor, &gamma_tensor, &beta_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
