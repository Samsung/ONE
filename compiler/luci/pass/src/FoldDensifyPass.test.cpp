/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldDensifyPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class FoldDensifyPassGraph : public luci::ConstantFoldingAddTestGraph
{
public:
  FoldDensifyPassGraph(std::initializer_list<uint32_t> shape)
    : luci::ConstantFoldingAddTestGraph(shape, loco::DataType::FLOAT32)
  {
    _densify = _g.nodes()->create<luci::CircleDensify>();
    _x = _g.nodes()->create<luci::CircleConst>();

    _densify->dtype(loco::DataType::FLOAT32);
    _x->dtype(loco::DataType::FLOAT32);

    _densify->shape(shape);
    _x->shape(shape);

    _densify->input(_x);

    _densify->name("densify");
    _x->name("x");
  }

  loco::Node *createFoldedPattern() override { return _densify; }

public:
  void fill_const_dense(void)
  {
    uint32_t num_elems = 1;
    for (uint32_t r = 0; r < _x->rank(); ++r)
      num_elems *= _x->dim(r).value();

    _x->size<loco::DataType::FLOAT32>(num_elems);
    for (uint32_t i = 0; i < num_elems; i++)
      _x->at<loco::DataType::FLOAT32>(i) = static_cast<float>(i + 1);
  }

  void fill_const_sparse(void)
  {
    // fill 4x4 of
    //  [[1 0 0 0]
    //   [0 2 0 0]
    //   [0 0 3 0]
    //   [0 0 0 4]]

    // values of 1.0, 2.0, 3.0, 4.0
    uint32_t udata[] = {0x3f800000, 0x40000000, 0x40400000, 0x40800000};
    float *fdata = reinterpret_cast<float *>(udata);

    _x->size<loco::DataType::FLOAT32>(4);
    for (uint32_t i = 0; i < 4; i++)
      _x->at<loco::DataType::FLOAT32>(i) = fdata[i];

    auto sparsityparam = std::make_unique<luci::SparsityParam>();
    sparsityparam->traversal_order = std::vector<int32_t>({0, 1});
    sparsityparam->block_map = std::vector<int32_t>({});

    auto dm0 = luci::DimMetaData(luci::DimensionType::DENSE, 4);

    std::vector<int32_t> as_vec = {0, 1, 2, 3, 4};
    std::vector<int32_t> ai_vec = {0, 1, 2, 3};
    auto as = luci::SparseIndexVector(luci::SparseIndexVectorType::I32, as_vec);
    auto ai = luci::SparseIndexVector(luci::SparseIndexVectorType::I32, ai_vec);
    auto dm1 = luci::DimMetaData(luci::DimensionType::SPARSE_CSR, 0, as, ai);
    sparsityparam->dim_metadata.emplace_back(dm0);
    sparsityparam->dim_metadata.emplace_back(dm1);

    _x->sparsityparam(std::move(sparsityparam));
  }

protected:
  luci::CircleDensify *_densify = nullptr;
  luci::CircleConst *_x = nullptr;
};

class FoldDensifyPassGraphTest : public FoldDensifyPassGraph, public ::testing::Test
{
public:
  FoldDensifyPassGraphTest() : FoldDensifyPassGraph({4, 4}) {}

  virtual void SetUp() { init(); }
};

} // namespace

TEST(FoldDensifyPassGraph, name)
{
  luci::FoldDensifyPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldDensifyPassGraphTest, no_sparsity_param_NEG)
{
  fill_const_dense();

  luci::FoldDensifyPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(FoldDensifyPassGraphTest, sparsity_param)
{
  fill_const_sparse();

  luci::FoldDensifyPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  EXPECT_EQ(2, folded_const->rank());
  EXPECT_EQ(4, folded_const->dim(0).value());
  EXPECT_EQ(4, folded_const->dim(1).value());
  EXPECT_EQ(16, folded_const->size<loco::DataType::FLOAT32>());
  for (int y = 0; y < 4; ++y)
  {
    for (int x = 0; x < 4; ++x)
    {
      float ovalue = folded_const->at<loco::DataType::FLOAT32>(y * 4 + x);
      float fvalue = 0.0;
      if (x == y)
      {
        // diagonal position
        fvalue = static_cast<float>(y + 1);
      }
      EXPECT_EQ(fvalue, ovalue);
    }
  }
}
