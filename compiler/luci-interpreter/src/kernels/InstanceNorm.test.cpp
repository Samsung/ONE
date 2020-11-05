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
#include "kernels/InstanceNorm.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;
TEST(InstanceNormTest, Simple)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 2, 2, 1}, {1, 1, 1, 1});
  Tensor gamma_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1});
  Tensor beta_tensor = makeInputTensor<DataType::FLOAT32>({1}, {2});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  InstanceNormParams params{};
  params.epsilon = 0.1f;
  params.activation = Activation::NONE;

  InstanceNorm kernel(&input_tensor, &gamma_tensor, &beta_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear({2, 2, 2, 2}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 1}));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
