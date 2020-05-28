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

#include "kernels/ArgMax.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(ArgMaxTest, Float)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<float> input_data{
      0.1, 0.9, 0.7, 0.3,
  };
  Shape dimension_shape{};
  std::vector<int32_t> dimension_data{
      3,
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor dimension_tensor = makeInputTensor<DataType::S32>(dimension_shape, dimension_data);
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Shape output_shape{1, 1, 1};

  ArgMaxParams params{};
  params.output_type = DataType::S64;
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();
  EXPECT_THAT(extractTensorData<int64_t>(output_tensor), ::testing::ElementsAreArray({1}));
  EXPECT_THAT(output_tensor.shape(), output_shape);
}

TEST(ArgMaxTest, Uint8)
{
  Shape input_shape{1, 1, 1, 4};
  std::vector<uint8_t> input_data{
      1, 9, 7, 3,
  };
  Shape dimension_shape{};
  std::vector<int32_t> dimension_data{
      3,
  };
  Tensor input_tensor = makeInputTensor<DataType::U8>(input_shape, input_data);
  Tensor dimension_tensor = makeInputTensor<DataType::S32>(dimension_shape, dimension_data);
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Shape output_shape{1, 1, 1};

  ArgMaxParams params{};
  params.output_type = DataType::S64;
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();
  EXPECT_THAT(extractTensorData<int64_t>(output_tensor), ::testing::ElementsAreArray({1}));
  EXPECT_THAT(output_tensor.shape(), output_shape);
}

TEST(ArgMaxTest, MultiDimensions)
{
  Shape input_shape{1, 1, 2, 4};
  std::vector<float> input_data{
      1, 2, 7, 8, 1, 9, 7, 3,
  };
  Shape dimension_shape{};
  std::vector<int32_t> dimension_data{
      3,
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor dimension_tensor = makeInputTensor<DataType::S32>(dimension_shape, dimension_data);
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Shape output_shape{1, 1, 2};

  ArgMaxParams params{};
  params.output_type = DataType::S64;
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();
  EXPECT_THAT(extractTensorData<int64_t>(output_tensor), ::testing::ElementsAreArray({3, 1}));
  EXPECT_THAT(output_tensor.shape(), output_shape);
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
