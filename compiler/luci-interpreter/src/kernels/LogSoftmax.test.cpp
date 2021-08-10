/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/LogSoftmax.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class LogSoftmaxTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<SimpleMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(LogSoftmaxTest, Float)
{
  Shape input_shape{2, 4};
  std::vector<float> input_data{
    0, -6, 2,  4, //
    3, -2, 10, 1, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LogSoftmax kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    -4.14297, -10.14297, -2.14297,   -.142971, //
    -7.00104, -12.00104, -.00104087, -9.00104, //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(LogSoftmaxTest, Uint8)
{
  float kMin = -10;
  float kMax = 10;
  float kLogSoftmaxQuantizedTolerance = 16. / 256;
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(kMin, kMax);
  std::vector<float> input_data{
    0, -6, 2,  4, //
    3, -2, 10, 1, //
  };
  Tensor input_tensor = makeInputTensor<DataType::U8>({2, 4}, quant_param.first, quant_param.second,
                                                      input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 16. / 256, 255);

  LogSoftmax kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    -4.14297, -10.14297, -2.14297,   -.142971, //
    -7.00104, -12.00104, -.00104087, -9.00104, //
  };
  std::vector<int32_t> ref_output_shape{2, 4};
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, kLogSoftmaxQuantizedTolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({189, 93, 221, 253, 142, 63, 255, 111}));
}

TEST_F(LogSoftmaxTest, InvalidInputOutputType_NEG)
{
  std::vector<float> input_data{
    0, -6, 2,  4, //
    3, -2, 10, 1, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 4}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 16. / 256, 255);

  LogSoftmax kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(LogSoftmaxTest, InvalidOutputQuantParam_NEG)
{
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-10, 10);
  std::vector<float> input_data{
    0, -6, 2,  4, //
    3, -2, 10, 1, //
  };
  Tensor input_tensor = makeInputTensor<DataType::U8>({2, 4}, quant_param.first, quant_param.second,
                                                      input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, 20. / 256, 255);

  LogSoftmax kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
