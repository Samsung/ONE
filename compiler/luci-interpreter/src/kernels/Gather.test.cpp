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

#include "kernels/Gather.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class GatherTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(GatherTest, Simple)
{
  std::vector<float> params_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<int32_t> indices_data{1, 0, 1, 5};
  std::vector<float> ref_output_data{2.f, 1.f, 2.f, 6.f};

  Tensor params_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 6}, params_data, _memory_manager.get());
  Tensor indices_tensor = makeInputTensor<DataType::S32>({4}, indices_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  GatherParams gparams;

  gparams.axis = 1;
  gparams.batch_dims = 0;

  Gather kernel(&params_tensor, &indices_tensor, &output_tensor, gparams);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 4}));
}

TEST_F(GatherTest, Simple_NEG)
{
  Tensor params_tensor = makeInputTensor<DataType::S32>({1}, {1}, _memory_manager.get());
  Tensor indices_tensor = makeInputTensor<DataType::S32>({1}, {0}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  GatherParams gparams;

  Gather kernel(&params_tensor, &indices_tensor, &output_tensor, gparams);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
