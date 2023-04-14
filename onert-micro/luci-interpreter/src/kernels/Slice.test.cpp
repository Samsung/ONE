/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#if 0
#include "kernels/Slice.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T> class SliceTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t, int8_t>;
TYPED_TEST_SUITE(SliceTest, DataTypes);

TYPED_TEST(SliceTest, SimpleTest)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  std::vector<TypeParam> input_data{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
  Shape input_shape{3, 2, 3, 1};
  std::vector<int32_t> begin_data{1, 0, 0, 0};
  Shape begin_shape{4};
  std::vector<int32_t> size_data{2, 1, -1, 1};
  Shape size_shape{4};
  std::vector<TypeParam> output_data{3, 3, 3, 5, 5, 5};
  std::vector<int32_t> output_shape{2, 1, 3, 1};

  Tensor input_tensor =
    makeInputTensor<getElementType<TypeParam>()>(input_shape, input_data, memory_manager.get());
  Tensor begin_tensor =
    makeInputTensor<DataType::S32>(begin_shape, begin_data, memory_manager.get());
  Tensor size_tensor = makeInputTensor<DataType::S32>(size_shape, size_data, memory_manager.get());

  Tensor output_tensor = makeOutputTensor(getElementType<TypeParam>());

  Slice kernel(&input_tensor, &begin_tensor, &size_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<TypeParam>(output_tensor),
              ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
