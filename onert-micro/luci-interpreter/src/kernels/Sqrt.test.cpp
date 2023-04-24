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

#include "kernels/Sqrt.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Sqrt kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(SqrtTest, SimpleSqrt)
{
  Check(
    /*input_shape=*/{1, 2, 4, 1}, /*output_shape=*/{1, 2, 4, 1},
    /*input_data=*/
    {
      0, 8, 2, 4,    //
      3, 7, 10, 0.3, //
    },
    /*output_data=*/
    {
      0.0, 2.8284271, 1.4142136, 2,                //
      1.7320508, 2.6457513, 3.1622777, 0.54772256, //
    });
}

TEST(SqrtTest, Input_Output_Type_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Sqrt kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(SqrtTest, Invalid_Input_Type_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Sqrt kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
