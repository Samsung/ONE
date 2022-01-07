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

#include "luci/Pass/FoldSlicePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class FoldSliceTestGraph : public luci::ConstantFoldingTestGraph
{
public:
  FoldSliceTestGraph(std::initializer_list<uint32_t> ishape, std::initializer_list<uint32_t> bshape,
                     std::initializer_list<uint32_t> sshape, std::initializer_list<uint32_t> oshape)
    : ConstantFoldingTestGraph(ishape, loco::DataType::FLOAT32)
  {
    _sinput = _g.nodes()->create<luci::CircleConst>();
    _begin = _g.nodes()->create<luci::CircleConst>();
    _size = _g.nodes()->create<luci::CircleConst>();
    _slice = _g.nodes()->create<luci::CircleSlice>();

    _sinput->dtype(loco::DataType::FLOAT32);
    _begin->dtype(loco::DataType::S32);
    _size->dtype(loco::DataType::S32);
    _slice->dtype(loco::DataType::FLOAT32);

    _sinput->shape(ishape);
    _begin->shape(bshape);
    _size->shape(sshape);
    _slice->shape(oshape);

    _slice->input(_sinput);
    _slice->begin(_begin);
    _slice->size(_size);

    _sinput->name("sinput");
    _begin->name("begin");
    _size->name("size");

    _output->from(_slice);
  }

protected:
  void init() final {}

protected:
  loco::Node *createFoldedPattern() final { return nullptr; }

protected:
  luci::CircleConst *getFoldedPattern() final
  {
    return loco::must_cast<luci::CircleConst *>(_output->from());
  }

protected:
  luci::CircleConst *_sinput = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_size = nullptr;
  luci::CircleSlice *_slice = nullptr;
};

class FoldSlicePassTest : public FoldSliceTestGraph, public ::testing::Test
{
public:
  FoldSlicePassTest() : FoldSliceTestGraph({3, 2, 3}, {3}, {3}, {1, 1, 3}) {}
};

} // namespace

TEST(FoldSliceTest, name)
{
  luci::FoldSlicePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldSlicePassTest, fold_slice)
{
  // TODO implement

  SUCCEED();
}

TEST_F(FoldSlicePassTest, fold_slice_NEG)
{
  // TODO implement

  SUCCEED();
}
