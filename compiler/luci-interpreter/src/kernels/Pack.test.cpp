/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Pack.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::vector<std::initializer_list<int32_t>> input_shapes,
           std::initializer_list<int32_t> output_shape, std::vector<std::vector<T>> input_datas,
           std::initializer_list<T> output_data, int32_t axis)
{
  constexpr DataType element_type = getElementType<T>();
  std::vector<const Tensor *> inputs(input_datas.size());
  std::vector<Tensor> tmp_inputs;
  for (int i = 0; i < input_datas.size(); i++)
  {
    tmp_inputs.push_back(makeInputTensor<element_type>(input_shapes[i], input_datas[i]));
  }
  for (int i = 0; i < input_datas.size(); i++)
  {
    inputs[i] = &tmp_inputs[i];
  }
  Tensor output_tensor = makeOutputTensor(element_type);
  PackParams params{};
  params.axis = axis;
  params.values_count = input_datas.size();
  Pack kernel(inputs, &output_tensor, params);

  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class PackTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(PackTest, DataTypes);

TYPED_TEST(PackTest, ThreeInputs)
{
  Check<TypeParam>(/*input_shapes=*/{{2}, {2}, {2}},
                   /*output_shape=*/{3, 2},
                   /*input_datas=*/
                   {{1, 4}, {2, 5}, {3, 6}},
                   /*output_data=*/
                   {1, 4, 2, 5, 3, 6}, 0);

  SUCCEED();
}

TEST(Pack, MismatchingInputValuesCount_NEG)
{
  std::vector<float> input1_data{1, 4};
  std::vector<float> input2_data{2, 5};
  std::vector<float> input3_data{3, 6};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2}, input1_data);
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2}, input2_data);
  Tensor input3_tensor = makeInputTensor<DataType::FLOAT32>({2}, input3_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  PackParams params{};
  {
    params.axis = 0;
    params.values_count = 2;

    Pack kernel({&input1_tensor, &input2_tensor, &input3_tensor}, &output_tensor, params);
    EXPECT_ANY_THROW(kernel.configure());
  }
}

TEST(Pack, InvalidInputAxis_NEG)
{
  std::vector<float> input1_data{1, 4};
  std::vector<float> input2_data{2, 5};
  std::vector<float> input3_data{3, 6};
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({2}, input1_data);
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({2}, input2_data);
  Tensor input3_tensor = makeInputTensor<DataType::FLOAT32>({2}, input3_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  PackParams params{};
  {
    params.axis = 2;
    params.values_count = 3;

    Pack kernel({&input1_tensor, &input2_tensor, &input3_tensor}, &output_tensor, params);
    EXPECT_ANY_THROW(kernel.configure());
  }
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
