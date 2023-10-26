/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Sin.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

#include <cmath>

namespace luci_interpreter
{
namespace kernels
{
namespace
{

#define PI 3.14159265358979323846

using namespace testing;

TEST(SinTest, Float)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 1, 3};
  std::vector<float> input_data{0.0f, PI / 3.0f, -PI / 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Sin kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> ref_output_shape{1, 1, 3};
  std::vector<float> ref_output_data{std::sin(0.0f), std::sin(PI / 3.0f), std::sin(-PI / 3.0f)};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(SinTest, InvalidDType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 1, 3};
  std::vector<int64_t> input_data{1l, 2l, 3l};
  Tensor input_tensor =
    makeInputTensor<DataType::S64>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Sin kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
