/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Shape.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ShapeTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

template <typename T> void runShapeKernel(loco::DataType dataType, IMemoryManager *memory_manager)
{
  Shape input_shape{1, 3, 1, 3, 5};

  Tensor input_tensor = Tensor(loco::DataType::FLOAT32, input_shape, {}, "");
  Tensor output_tensor = makeOutputTensor(dataType);

  ShapeParams params{};
  params.out_type = dataType;

  ShapeKernel kernel(&input_tensor, &output_tensor, params);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<T> ref_output_data{1, 3, 1, 3, 5};
  EXPECT_THAT(extractTensorData<T>(output_tensor), ref_output_data);

  std::vector<int32_t> ref_output_shape{5};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(ShapeTest, OutTypeInt)
{

  // Run for int32_t output
  runShapeKernel<int32_t>(loco::DataType::S32, _memory_manager.get());
  // Run for int64_t output
  runShapeKernel<int64_t>(loco::DataType::S64, _memory_manager.get());

  SUCCEED();
}

TEST_F(ShapeTest, Invalid_Output_Type_NEG)
{
  Shape input_shape{1, 3};

  Tensor input_tensor = Tensor(loco::DataType::FLOAT32, input_shape, {}, "");
  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  ShapeParams params{};
  params.out_type = loco::DataType::FLOAT32;

  ShapeKernel kernel(&input_tensor, &output_tensor, params);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
