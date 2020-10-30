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

#include "Sparsifier.h"

#include <vector>

#include <gtest/gtest.h>

TEST(SparsifierTest, NoBlockDenseDense)
{
  const std::vector<int32_t> dense_tensor_data = {1, 2, 3, 4, 5, 6};
  const std::vector<int32_t> dense_shape = {2, 3};
  const std::vector<int32_t> traversal_order = {0, 1};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::DENSE};
  luci::Sparsifier<int32_t> sparsifier(dense_shape, traversal_order, format);

  sparsifier.DenseToSparse(dense_tensor_data.data());

  const auto dim_metadata = sparsifier.GetDimMetadata();
  const std::vector<int32_t> expected_dm0 = {2};
  const std::vector<int32_t> expected_dm1 = {3};

  EXPECT_EQ(/*dense_shape[0]=*/expected_dm0, dim_metadata[0]);
  EXPECT_EQ(/*dense_shape[1]=*/expected_dm1, dim_metadata[2]);

  const auto sparsed_data = sparsifier.GetData();
  const std::vector<int32_t> expected_data = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(expected_data, sparsed_data);
}

TEST(SparsifierTest, NoBlockDenseCSR)
{
  const std::vector<int32_t> dense_tensor_data = {0, 1, 2, 3, 0, 5, 0, 0, 0};
  const std::vector<int32_t> dense_shape = {3, 3};
  const std::vector<int32_t> traversal_order = {0, 1};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::SPARSE_CSR};
  luci::Sparsifier<int32_t> sparsifier(dense_shape, traversal_order, format);
  sparsifier.DenseToSparse(dense_tensor_data.data());

  const auto dim_metadata = sparsifier.GetDimMetadata();
  const std::vector<int32_t> expected_dm0 = {3};
  const std::vector<int32_t> expected_dm1 = {};
  const std::vector<int32_t> expected_dm2 = {0, 2, 4, 4};
  const std::vector<int32_t> expected_dm3 = {1, 2, 0, 2};

  EXPECT_EQ(expected_dm0, dim_metadata[0]);
  EXPECT_EQ(expected_dm1, dim_metadata[1]);
  EXPECT_EQ(expected_dm2, dim_metadata[2]);
  EXPECT_EQ(expected_dm3, dim_metadata[3]);

  const auto data = sparsifier.GetData();
  const std::vector<int32_t> expected_data = {1, 2, 3, 5};
  EXPECT_EQ(expected_data, data);
}

TEST(SparsifierTest, BlockDenseDense)
{
  const std::vector<float> dense_tensor_data = {1.1, 2.2,  3.3,  4.4,  5.5,  6.6,  7.7,  8.8,
                                                9.9, 10.0, 11.1, 12.2, 13.3, 14.4, 15.5, 16.6};
  const std::vector<int32_t> dense_shape = {4, 4};
  const std::vector<int32_t> traversal_order = {0, 1, 2, 3};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::DENSE};
  const std::vector<int32_t> block_size = {2, 2};
  const std::vector<int32_t> block_map = {0, 1};
  luci::Sparsifier<float> sparsifier(dense_shape, traversal_order, format, block_size, block_map);
  sparsifier.DenseToSparse(dense_tensor_data.data());

  const auto dim_metadata = sparsifier.GetDimMetadata();
  const std::vector<int32_t> expected_dm0 = {2};
  const std::vector<int32_t> expected_dm1 = {};
  EXPECT_EQ(expected_dm0, dim_metadata[0]);
  EXPECT_EQ(expected_dm1, dim_metadata[1]);
  EXPECT_EQ(expected_dm0, dim_metadata[2]);
  EXPECT_EQ(expected_dm1, dim_metadata[3]);
  EXPECT_EQ(expected_dm0, dim_metadata[4]);
  EXPECT_EQ(expected_dm1, dim_metadata[5]);
  EXPECT_EQ(expected_dm0, dim_metadata[6]);
  EXPECT_EQ(expected_dm1, dim_metadata[7]);

  const auto data = sparsifier.GetData();
  const std::vector<float> expected_data = {1.1, 2.2,  5.5,  6.6,  3.3,  4.4,  7.7,  8.8,
                                            9.9, 10.0, 13.3, 14.4, 11.1, 12.2, 15.5, 16.6};
  EXPECT_EQ(expected_data, data);
}

TEST(SparsifierTest, BlockDenseSparse)
{
  const std::vector<int32_t> dense_tensor_data = {1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0};
  const std::vector<int32_t> dense_shape = {4, 4};
  const std::vector<int32_t> traversal_order = {0, 1, 2, 3};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::SPARSE_CSR};
  const std::vector<int32_t> block_size = {2, 2};
  const std::vector<int32_t> block_map = {0, 1};
  luci::Sparsifier<int32_t> sparsifier(dense_shape, traversal_order, format, block_size, block_map);
  sparsifier.DenseToSparse(dense_tensor_data.data());

  const auto dim_metadata = sparsifier.GetDimMetadata();
  const std::vector<int32_t> expected_dm0 = {2};
  const std::vector<int32_t> expected_dm1 = {};
  const std::vector<int32_t> expected_dm2 = {0, 1, 2};
  const std::vector<int32_t> expected_dm3 = {0, 0};
  EXPECT_EQ(expected_dm0, dim_metadata[0]);
  EXPECT_EQ(expected_dm1, dim_metadata[1]);
  EXPECT_EQ(expected_dm2, dim_metadata[2]);
  EXPECT_EQ(expected_dm3, dim_metadata[3]);
  EXPECT_EQ(expected_dm0, dim_metadata[4]);
  EXPECT_EQ(expected_dm1, dim_metadata[5]);
  EXPECT_EQ(expected_dm0, dim_metadata[6]);
  EXPECT_EQ(expected_dm1, dim_metadata[7]);

  const auto data = sparsifier.GetData();
  const std::vector<int32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(expected_data, data);
}

TEST(SparsifierTest, BlockDenseSparse_2)
{
  const std::vector<int32_t> dense_tensor_data = {0,  4, 8,  1,  5, 9,  2,  6, 10, 3,  7, 11,
                                                  12, 0, 20, 13, 0, 21, 14, 0, 22, 15, 0, 23};
  const std::vector<int32_t> dense_shape = {8, 3};
  const std::vector<int32_t> traversal_order = {0, 1, 2, 3};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::SPARSE_CSR};
  const std::vector<int32_t> block_size = {4, 1};
  const std::vector<int32_t> block_map = {0, 1};
  luci::Sparsifier<int32_t> sparsifier(dense_shape, traversal_order, format, block_size, block_map);
  sparsifier.DenseToSparse(dense_tensor_data.data());

  const auto dim_metadata = sparsifier.GetDimMetadata();
  const std::vector<int32_t> expected_dm0 = {2};
  const std::vector<int32_t> expected_dm1 = {};
  const std::vector<int32_t> expected_dm2 = {0, 3, 5};
  const std::vector<int32_t> expected_dm3 = {0, 1, 2, 0, 2};
  const std::vector<int32_t> expected_dm4 = {4};
  const std::vector<int32_t> expected_dm6 = {1};
  EXPECT_EQ(expected_dm0, dim_metadata[0]);
  EXPECT_EQ(expected_dm1, dim_metadata[1]);
  EXPECT_EQ(expected_dm2, dim_metadata[2]);
  EXPECT_EQ(expected_dm3, dim_metadata[3]);
  EXPECT_EQ(expected_dm4, dim_metadata[4]);
  EXPECT_EQ(expected_dm1, dim_metadata[5]);
  EXPECT_EQ(expected_dm6, dim_metadata[6]);
  EXPECT_EQ(expected_dm1, dim_metadata[7]);

  const auto data = sparsifier.GetData();
  const std::vector<int32_t> expected_data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                              10, 11, 12, 13, 14, 15, 20, 21, 22, 23};
  EXPECT_EQ(expected_data, data);
}

TEST(SparsifierTest, WrongTraversalOrderRank_NEG)
{
  const std::vector<int32_t> dense_tensor_data = {0,  4, 8,  1,  5, 9,  2,  6, 10, 3,  7, 11,
                                                  12, 0, 20, 13, 0, 21, 14, 0, 22, 15, 0, 23};
  const std::vector<int32_t> dense_shape = {8, 3};
  const std::vector<int32_t> traversal_order = {0, 1};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::DENSE,
                                                   luci::DimensionType::SPARSE_CSR};
  const std::vector<int32_t> block_size = {4, 1};
  const std::vector<int32_t> block_map = {0, 1};
  luci::Sparsifier<int32_t> sparsifier(dense_shape, traversal_order, format, block_size, block_map);
  EXPECT_THROW(sparsifier.DenseToSparse(dense_tensor_data.data()), std::out_of_range);
}

TEST(SparsifierTest, WrongFormatRank_NEG)
{
  const std::vector<int32_t> dense_tensor_data = {0,  4, 8,  1,  5, 9,  2,  6, 10, 3,  7, 11,
                                                  12, 0, 20, 13, 0, 21, 14, 0, 22, 15, 0, 23};
  const std::vector<int32_t> dense_shape = {8, 3};
  const std::vector<int32_t> traversal_order = {0, 1, 2, 3};
  const std::vector<luci::DimensionType> format = {luci::DimensionType::SPARSE_CSR};
  const std::vector<int32_t> block_size = {4, 1};
  const std::vector<int32_t> block_map = {0, 1};
  EXPECT_THROW(
      luci::Sparsifier<int32_t>(dense_shape, traversal_order, format, block_size, block_map),
      std::out_of_range);
}
