/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Sign.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

#include <cmath>

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SignTest, S32)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  // 0, +, -
  Shape input_shape{1, 1, 3};
  std::vector<int32_t> input_data{0, 2, -3};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Sign kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  // shape check
  std::vector<int32_t> ref_output_shape{1, 1, 3};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));

  // data check
  auto out = extractTensorData<int32_t>(output_tensor);

  EXPECT_THAT(std::vector<int32_t>(out.begin(), out.end()), ::testing::ElementsAre(0, 1, -1));
}

TEST(SignTest, Float32)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  // 0, +, -, NaN
  Shape input_shape{1, 1, 4};
  std::vector<float> input_data{0.0f, 2.0f, -3.0f, std::numeric_limits<float>::quiet_NaN()};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Sign kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  // shape check
  std::vector<int32_t> ref_output_shape{1, 1, 4};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));

  // data check
  auto out = extractTensorData<float>(output_tensor);

  // first 3 are deterministic
  std::vector<float> ref_first3{0.0f, 1.0f, -1.0f};
  EXPECT_THAT(std::vector<float>(out.begin(), out.begin() + 3), FloatArrayNear(ref_first3));

  // NaN should stay NaN
  EXPECT_TRUE(std::isnan(out[3]));
}

TEST(SignTest, Float64)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  // 0, +, -, NaN
  Shape input_shape{1, 1, 4};
  std::vector<double> input_data{0.0, 2.0, -3.0, std::numeric_limits<double>::quiet_NaN()};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT64>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT64);

  Sign kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  // shape check
  std::vector<int32_t> ref_output_shape{1, 1, 4};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));

  // data check
  auto out = extractTensorData<double>(output_tensor);

  // first 3 are deterministic
  std::vector<double> ref_first3{0.0, 1.0, -1.0};
  EXPECT_THAT(std::vector<double>(out.begin(), out.begin() + 3), DoubleArrayNear(ref_first3));

  // NaN should stay NaN
  EXPECT_TRUE(std::isnan(out[3]));
}

TEST(SignTest, InvalidDType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 1, 3};
  std::vector<int64_t> input_data{1l, 2l, 3l};

  Tensor input_tensor =
    makeInputTensor<DataType::S64>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Sign kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
