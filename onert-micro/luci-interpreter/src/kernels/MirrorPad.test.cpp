/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/MirrorPad.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class MirrorPadTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  void Execute(const Tensor &input, const Tensor &padding, Tensor &output, MirrorPadMode mode)
  {
    MirrorPadParams params{};
    params.mode = mode;

    MirrorPad kernel(&input, &padding, &output, params);
    kernel.configure();
    _memory_manager->allocate_memory(output);
    kernel.execute();
  }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(MirrorPadTest, FloatReflect)
{
  Shape input_shape = {1, 2, 2, 1};
  Shape padding_shape = {4, 2};

  std::vector<float> input_data{1.0f, 2.0f,  //
                                3.0f, 4.0f}; //
  std::vector<int> padding_data{0, 0, 2, 1, 1, 2, 0, 0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor padding_tensor =
    makeInputTensor<DataType::S32>(padding_shape, padding_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::REFLECT);

  std::vector<float> ref_output_data{2.0f, 1.0f, 2.0f, 1.0f, 2.0f,  //
                                     4.0f, 3.0f, 4.0f, 3.0f, 4.0f,  //
                                     2.0f, 1.0f, 2.0f, 1.0f, 2.0f,  //
                                     4.0f, 3.0f, 4.0f, 3.0f, 4.0f,  //
                                     2.0f, 1.0f, 2.0f, 1.0f, 2.0f}; //
  std::initializer_list<int32_t> ref_output_shape{1, 5, 5, 1};

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MirrorPadTest, FloatSymmetric)
{
  Shape input_shape = {1, 2, 2, 1};
  Shape padding_shape = {4, 2};

  std::vector<float> input_data{1.0f, 2.0f,  //
                                3.0f, 4.0f}; //
  std::vector<int> padding_data{0, 0, 2, 1, 1, 2, 0, 0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor padding_tensor =
    makeInputTensor<DataType::S32>(padding_shape, padding_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::SYMMETRIC);

  std::vector<float> ref_output_data{3.0, 3.0, 4.0, 4.0, 3.0,  //
                                     1.0, 1.0, 2.0, 2.0, 1.0,  //
                                     1.0, 1.0, 2.0, 2.0, 1.0,  //
                                     3.0, 3.0, 4.0, 4.0, 3.0,  //
                                     3.0, 3.0, 4.0, 4.0, 3.0}; //
  std::initializer_list<int32_t> ref_output_shape{1, 5, 5, 1};

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MirrorPadTest, FloatSymmetric2Dim)
{
  Shape input_shape = {3, 1};
  Shape padding_shape = {2, 2};

  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  std::vector<int> padding_data{1, 2, 0, 0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor padding_tensor =
    makeInputTensor<DataType::S32>(padding_shape, padding_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::SYMMETRIC);

  std::vector<float> ref_output_data{1.0, 1.0, 2.0, 3.0, 3.0, 2.0};
  std::initializer_list<int32_t> ref_output_shape{6, 1};

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MirrorPadTest, Uint8Reflect)
{
  Shape input_shape = {1, 2, 3, 1};
  Shape padding_shape = {4, 2};

  float quant_tolerance = getTolerance(0.0f, 6.0f, 255);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(0.0f, 6.0f);

  std::vector<float> input_data{1.0f, 2.0f, 3.0f,  //
                                4.0f, 5.0f, 6.0f}; //
  std::vector<int> padding_data{0, 0, 2, 1, 1, 3, 0, 0};

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    input_shape, quant_param.first, quant_param.second, input_data, _memory_manager.get());

  Tensor padding_tensor =
    makeInputTensor<DataType::S32>(padding_shape, padding_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::REFLECT);

  std::vector<float> ref_output_data{
    3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, //
    6.0f, 4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f, //
    3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, //
    6.0f, 4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f, //
    3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, //
  };
  std::initializer_list<int32_t> ref_output_shape{1, 5, 7, 1};

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, quant_tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MirrorPadTest, Uint8Symmetric)
{
  Shape input_shape = {1, 2, 3, 1};
  Shape padding_shape = {4, 2};

  float quant_tolerance = getTolerance(0.0f, 6.0f, 255);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(0.0f, 6.0f);

  std::vector<float> input_data{1.0f, 2.0f, 3.0f,  //
                                4.0f, 5.0f, 6.0f}; //
  std::vector<int> padding_data{0, 0, 2, 1, 1, 3, 0, 0};

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    input_shape, quant_param.first, quant_param.second, input_data, _memory_manager.get());

  Tensor padding_tensor =
    makeInputTensor<DataType::S32>(padding_shape, padding_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::SYMMETRIC);

  std::vector<float> ref_output_data{
    4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, //
    1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f, //
    1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f, //
    4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, //
    4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, //
  };
  std::initializer_list<int32_t> ref_output_shape{1, 5, 7, 1};

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, quant_tolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MirrorPadTest, UnsupportedDim_NEG)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 1, 1}, {1.0f}, _memory_manager.get());
  Tensor padding_tensor =
    makeInputTensor<DataType::S32>({5, 2}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  EXPECT_ANY_THROW(Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::REFLECT));
}

TEST_F(MirrorPadTest, InvalidInputType_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1}, _memory_manager.get());
  Tensor padding_tensor = makeInputTensor<DataType::S32>({1, 2}, {0, 0}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  EXPECT_ANY_THROW(Execute(input_tensor, padding_tensor, output_tensor, MirrorPadMode::REFLECT));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
