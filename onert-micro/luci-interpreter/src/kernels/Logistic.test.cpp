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
// TODO enable it
#if 0
#include "kernels/Logistic.h"
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
           std::initializer_list<float> input_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<getElementType<T>()>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(getElementType<T>());

  Logistic kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <>
void Check<uint8_t>(std::initializer_list<int32_t> input_shape,
                    std::initializer_list<int32_t> output_shape,
                    std::initializer_list<float> input_data,
                    std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  std::pair<float, int32_t> input_quant_param =
    quantizationParams<uint8_t>(std::min(input_data), std::max(input_data));
  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1. / 256, 0);

  Logistic kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale() * 2));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class LogisticTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(LogisticTest, DataTypes);

TYPED_TEST(LogisticTest, Simple)
{
  Check<TypeParam>(
    {89}, {89},
    {-10.0000000000, -9.7727272727, -9.5454545455, -9.3181818182, -9.0909090909, -8.8636363636,
     -8.6363636364,  -8.4090909091, -8.1818181818, -7.9545454545, -7.7272727273, -7.5000000000,
     -7.2727272727,  -7.0454545455, -6.8181818182, -6.5909090909, -6.3636363636, -6.1363636364,
     -5.9090909091,  -5.6818181818, -5.4545454545, -5.2272727273, -5.0000000000, -4.7727272727,
     -4.5454545455,  -4.3181818182, -4.0909090909, -3.8636363636, -3.6363636364, -3.4090909091,
     -3.1818181818,  -2.9545454545, -2.7272727273, -2.5000000000, -2.2727272727, -2.0454545455,
     -1.8181818182,  -1.5909090909, -1.3636363636, -1.1363636364, -0.9090909091, -0.6818181818,
     -0.4545454545,  -0.2272727273, 0.0000000000,  0.2272727273,  0.4545454545,  0.6818181818,
     0.9090909091,   1.1363636364,  1.3636363636,  1.5909090909,  1.8181818182,  2.0454545455,
     2.2727272727,   2.5000000000,  2.7272727273,  2.9545454545,  3.1818181818,  3.4090909091,
     3.6363636364,   3.8636363636,  4.0909090909,  4.3181818182,  4.5454545455,  4.7727272727,
     5.0000000000,   5.2272727273,  5.4545454545,  5.6818181818,  5.9090909091,  6.1363636364,
     6.3636363636,   6.5909090909,  6.8181818182,  7.0454545455,  7.2727272727,  7.5000000000,
     7.7272727273,   7.9545454545,  8.1818181818,  8.4090909091,  8.6363636364,  8.8636363636,
     9.0909090909,   9.3181818182,  9.5454545455,  9.7727272727,  10.0000000000},
    {0.0000453979, 0.0000569815, 0.0000715205, 0.0000897689, 0.0001126729, 0.0001414198,
     0.0001774998, 0.0002227827, 0.0002796147, 0.0003509396, 0.0004404502, 0.0005527786,
     0.0006937345, 0.0008706021, 0.0010925128, 0.0013709094, 0.0017201256, 0.0021581065,
     0.0027073042, 0.0033957870, 0.0042586071, 0.0053394826, 0.0066928509, 0.0083863576,
     0.0105038445, 0.0131488902, 0.0164489307, 0.0205599431, 0.0256715863, 0.0320125562,
     0.0398556989, 0.0495221198, 0.0613831074, 0.0758581800, 0.0934070047, 0.1145124805,
     0.1396521834, 0.1692560327, 0.2036499335, 0.2429886272, 0.2871859014, 0.3358556241,
     0.3882805886, 0.4434251301, 0.5000000000, 0.5565748699, 0.6117194114, 0.6641443759,
     0.7128140986, 0.7570113728, 0.7963500665, 0.8307439673, 0.8603478166, 0.8854875195,
     0.9065929953, 0.9241418200, 0.9386168926, 0.9504778802, 0.9601443011, 0.9679874438,
     0.9743284137, 0.9794400569, 0.9835510693, 0.9868511098, 0.9894961555, 0.9916136424,
     0.9933071491, 0.9946605174, 0.9957413929, 0.9966042130, 0.9972926958, 0.9978418935,
     0.9982798744, 0.9986290906, 0.9989074872, 0.9991293979, 0.9993062655, 0.9994472214,
     0.9995595498, 0.9996490604, 0.9997203853, 0.9997772173, 0.9998225002, 0.9998585802,
     0.9998873271, 0.9999102311, 0.9999284795, 0.9999430185, 0.9999546021});
}

TEST(LogisticTest, IvalidInputOutputType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Shape input_shape = {1};
  std::vector<float> input_data{10};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1. / 256, 0);

  Logistic kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(LogisticTest, IvalidQuantParam_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape = {2};
  std::vector<float> input_data{-10, 10};
  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(-10, 10);
  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1. / 255, 0);

  Logistic kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
