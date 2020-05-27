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

#include "kernels/Pad.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(Pad, Float)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6};
  std::vector<int32_t> paddings_data{1, 0, 0, 2, 0, 3, 0, 0};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 2, 3, 1}, input_data);
  Tensor paddings_tensor = makeInputTensor<DataType::S32>({4, 2}, paddings_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pad kernel(&input_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5,
                                     6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
