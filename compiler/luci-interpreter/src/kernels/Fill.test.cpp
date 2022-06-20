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

#include "kernels/Fill.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class FillTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(FillTest, FillInt32)
{
  Shape dims_shape{2};

  std::vector<int32_t> dims_data = {2, 3};
  std::vector<int32_t> value_data = {-11};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value =
    makeInputTensor<loco::DataType::S32>(/*scalar*/ {}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::S32);

  Fill kernel(&dims, &value, &output_tensor);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> ref_output_data{-11, -11, -11, -11, -11, -11};

  std::vector<int32_t> ref_output_shape{2, 3};
  EXPECT_THAT(extractTensorData<int32_t>(output_tensor), ref_output_data);
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FillTest, FillFloat)
{
  Shape dims_shape{3};

  std::vector<int32_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value =
    makeInputTensor<loco::DataType::FLOAT32>(/*scalar*/ {}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{5, 5, 5, 5, 5, 5, 5, 5};

  std::vector<int32_t> ref_output_shape{2, 2, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor), ref_output_data);
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FillTest, Invalid_Input_Shape_NEG)
{
  Shape dims_shape{1, 3};

  std::vector<int32_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value =
    makeInputTensor<loco::DataType::FLOAT32>(/*scalar*/ {}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(FillTest, Invalid_Value_Shape_NEG)
{
  Shape dims_shape{3};

  std::vector<int32_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value = makeInputTensor<loco::DataType::FLOAT32>({1}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
