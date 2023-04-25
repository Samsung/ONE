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

#include "kernels/BatchMatMul.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class BatchMatMulTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(BatchMatMulTest, Float)
{
  std::vector<float> lhs_data = {1, 2, 3, 4, 5, 6};
  std::vector<float> rhs_data = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 3}, lhs_data, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 3, 4}, rhs_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  kernel.configure();
  _memory_manager->allocate_memory(lhs_scratch);
  _memory_manager->allocate_memory(rhs_scratch);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4}));
}

TEST_F(BatchMatMulTest, Float_SimpleRHSAdjoint)
{
  std::vector<float> lhs_data = {1, 2, 3, 4, 5, 6};
  std::vector<float> rhs_data = {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18};
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 3}, lhs_data, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 4, 3}, rhs_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = true;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  kernel.configure();
  _memory_manager->allocate_memory(lhs_scratch);
  _memory_manager->allocate_memory(rhs_scratch);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4}));
}

TEST_F(BatchMatMulTest, Float_SimpleLHSAdjoint)
{
  std::vector<float> lhs_data = {1, 4, 2, 5, 3, 6};
  std::vector<float> rhs_data = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 3, 2}, lhs_data, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 3, 4}, rhs_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = true;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  kernel.configure();
  _memory_manager->allocate_memory(lhs_scratch);
  _memory_manager->allocate_memory(rhs_scratch);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4}));
}

TEST_F(BatchMatMulTest, Float_BatchSizeTwo)
{
  std::vector<float> lhs_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> rhs_data = {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 2, 3}, lhs_data, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3, 4}, rhs_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  kernel.configure();
  _memory_manager->allocate_memory(lhs_scratch);
  _memory_manager->allocate_memory(rhs_scratch);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({74., 80., 86., 92., 173., 188., 203., 218., 560., 584., 608., 632.,
                              767., 800., 833., 866.}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 2, 4}));
}

TEST_F(BatchMatMulTest, Float_DiffBatch)
{
  std::vector<float> lhs_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> rhs_data = {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 1, 6}, lhs_data, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 6, 4}, rhs_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  kernel.configure();
  _memory_manager->allocate_memory(lhs_scratch);
  _memory_manager->allocate_memory(rhs_scratch);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({427., 448., 469., 490., 1039., 1096., 1153., 1210.}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 1, 4}));
}

TEST_F(BatchMatMulTest, Invalid_Shape_NEG)
{
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 2}, {1, 2, 3, 4}, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 3, 2}, {5, 6, 7, 8, 9, 10}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(BatchMatMulTest, Invalid_Batch_NEG)
{
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 1, 3}, {1, 2, 3, 4, 5, 6}, _memory_manager.get());
  Tensor rhs_tensor = makeInputTensor<DataType::FLOAT32>({3, 3, 1}, {5, 6, 7, 8, 9, 10, 11, 12, 13},
                                                         _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(BatchMatMulTest, Invalid_Rank_NEG)
{
  Tensor lhs_tensor = makeInputTensor<DataType::FLOAT32>({4}, {1, 2, 3, 4}, _memory_manager.get());
  Tensor rhs_tensor = makeInputTensor<DataType::FLOAT32>({1, 4, 2}, {5, 6, 7, 8, 9, 10, 11, 12},
                                                         _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(BatchMatMulTest, Invalid_Rank2_NEG)
{
  Tensor lhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 1, 4}, {1, 2, 3, 4}, _memory_manager.get());
  Tensor rhs_tensor = makeInputTensor<DataType::FLOAT32>({1, 4, 2}, {5, 6, 7, 8, 9, 10, 11, 12},
                                                         _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::FLOAT32, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(BatchMatMulTest, TypeMisMatch_NEG)
{
  Tensor lhs_tensor =
    makeInputTensor<DataType::U8>({1, 2, 3}, {1, 2, 3, 4, 5, 6}, _memory_manager.get());
  Tensor rhs_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 3, 2}, {5, 6, 7, 8, 9, 10}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor lhs_scratch(DataType::U8, Shape({}), {}, "");
  Tensor rhs_scratch(DataType::FLOAT32, Shape({}), {}, "");

  BatchMatMulParams params;
  params.adj_x = false;
  params.adj_y = false;

  BatchMatMul kernel(&lhs_tensor, &rhs_tensor, &output_tensor, &lhs_scratch, &rhs_scratch, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
