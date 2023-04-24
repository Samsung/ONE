/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Gelu.h"
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
           std::initializer_list<float> input_data, std::initializer_list<float> output_data,
           bool approximate)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<float>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  GeluParams params{};
  params.approximate = approximate;

  Gelu kernel(&input_tensor, &output_tensor, params);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
}

class GeluTest : public ::testing::Test
{
};

TEST_F(GeluTest, Simple)
{
  Check(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 3},
        /*input_data=*/
        {
          0.0f, 1.0f, 3.0f,   // Row 1
          1.0f, -1.0f, -2.0f, // Row 2
        },
        /*output_data=*/
        {
          0.0f, 0.841345f, 2.99595f,          // Row 1
          0.841345f, -0.158655f, -0.0455003f, // Row 2
        },
        /*approximate=*/false);

  SUCCEED();
}

TEST_F(GeluTest, Approximate)
{
  Check(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 3},
        /*input_data=*/
        {
          0.0f, 1.0f, 3.0f,   // Row 1
          1.0f, -1.0f, -2.0f, // Row 2
        },
        /*output_data=*/
        {
          0.0f, 0.841192f, 2.99636f,          // Row 1
          0.841192f, -0.158808f, -0.0454023f, // Row 2
        },
        /*approximate=*/true);

  SUCCEED();
}

TEST_F(GeluTest, DifferentInOutType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 3},
                                                           {
                                                             0.0f, 1.0f, 3.0f,   // Row 1
                                                             1.0f, -1.0f, -2.0f, // Row 2
                                                           },
                                                           memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  GeluParams params{};
  params.approximate = false;

  Gelu kernel(&input_tensor, &output_tensor, params);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
