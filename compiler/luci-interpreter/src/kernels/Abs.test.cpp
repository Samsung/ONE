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

#include "kernels/Abs.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<T> input_data, std::initializer_list<T> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  Abs kernel(&input_tensor, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(AbsTest, FloatSimple)
{
  Check<float>(/*input_shape=*/{2, 3},
               /*output_shape=*/{2, 3},
               /*input_data=*/
               {
                 0.0f, -1.0f, 3.0f,  // Row 1
                 1.0f, -1.0f, -2.0f, // Row 2
               },
               /*output_data=*/
               {
                 0.0f, 1.0f, 3.0f, // Row 1
                 1.0f, 1.0f, 2.0f, // Row 2
               });

  SUCCEED();
}

TEST(AbsTest, Type_Mismatch_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<loco::DataType::S32>({3}, {1, -3, 2}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Abs kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
