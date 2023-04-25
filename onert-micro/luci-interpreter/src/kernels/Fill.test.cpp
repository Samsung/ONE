/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Fill.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class FillTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

template <typename T, DataType DT> void runFillIntKernel(IMemoryManager *memory_manager)
{
  Shape dims_shape{2};

  std::vector<int32_t> dims_data = {2, 3};
  std::vector<T> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, memory_manager);
  Tensor value = makeInputTensor<DT>(/*scalar*/ {}, value_data, memory_manager);

  Tensor output_tensor = makeOutputTensor(DT);

  Fill kernel(&dims, &value, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<T> ref_output_data{5, 5, 5, 5, 5, 5};
  EXPECT_THAT(extractTensorData<T>(output_tensor), ref_output_data);

  std::vector<int32_t> ref_output_shape{2, 3};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

template <DataType DT> void runFillQuantIntKernel(IMemoryManager *memory_manager)
{
  Shape dims_shape{2};

  std::vector<int32_t> dims_data = {2, 3};
  std::vector<float> value_data = {5};

  int32_t zero_point = 0;

  if (DT == loco::DataType::S8)
    zero_point = 1;

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, memory_manager);
  Tensor value = makeInputTensor<DT>(/*scalar*/ {}, /*scale*/ 0.25, /*zero_point*/ zero_point,
                                     value_data, memory_manager);

  Tensor output_tensor = makeOutputTensor(DT, /*scale*/ 0.25, /*zero_point*/ zero_point);

  Fill kernel(&dims, &value, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{5, 5, 5, 5, 5, 5};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));

  std::vector<int32_t> ref_output_shape{2, 3};
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FillTest, FillInt)
{
  // Run for int32_t input
  runFillIntKernel<int32_t, loco::DataType::S32>(_memory_manager.get());
  // Run for int64_t input
  runFillIntKernel<int64_t, loco::DataType::S64>(_memory_manager.get());
  // Run for int8_t input
  runFillQuantIntKernel<loco::DataType::S8>(_memory_manager.get());
  // Run for int16_t input
  runFillQuantIntKernel<loco::DataType::S16>(_memory_manager.get());

  SUCCEED();
}

TEST_F(FillTest, FillFloat)
{
  Shape dims_shape{3};

  std::vector<int64_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S64>(dims_shape, dims_data, _memory_manager.get());
  Tensor value =
    makeInputTensor<loco::DataType::FLOAT32>(/*scalar*/ {}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);

  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{5, 5, 5, 5, 5, 5, 5, 5};

  std::vector<int32_t> ref_output_shape{2, 2, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor), ref_output_data);
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FillTest, Invalid_Input_Shape_NEG)
{
  Shape dims_shape{1, 3};

  std::vector<int32_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value =
    makeInputTensor<loco::DataType::FLOAT32>(/*scalar*/ {}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(FillTest, Invalid_Value_Shape_NEG)
{
  Shape dims_shape{3};

  std::vector<int32_t> dims_data = {2, 2, 2};
  std::vector<float> value_data = {5};

  Tensor dims = makeInputTensor<loco::DataType::S32>(dims_shape, dims_data, _memory_manager.get());
  Tensor value = makeInputTensor<loco::DataType::FLOAT32>({1}, value_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(loco::DataType::FLOAT32);

  Fill kernel(&dims, &value, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
