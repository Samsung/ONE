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

#include "LayerInfoMap.h"

#include <luci/IR/CircleNode.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

class SoftmaxTestGraph : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _softmax = g()->nodes()->create<luci::CircleSoftmax>();
    {
      _softmax->logits(input());
      _softmax->beta(0.1);
      _softmax->name("test");
    }
    output()->from(_softmax);
  }

private:
  luci::CircleSoftmax *_softmax = nullptr;
};

class SplitAddTestGraph : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({6, 1, 2}, {3, 1, 2});
    _split_dim = g()->nodes()->create<luci::CircleConst>();
    {
      _split_dim->rank(1);
      _split_dim->dtype(loco::DataType::S32);
      _split_dim->size<loco::DataType::S32>(1);
      _split_dim->at<loco::DataType::S32>(0);
      _split_dim->shape({1});
      _split_dim->name("split_dim");
    }

    _split = g()->nodes()->create<luci::CircleSplit>();
    {
      _split->input(input());
      _split->num_split(2);
      _split->split_dim(_split_dim);
      _split->name("split0");
    }

    _split_out_1 = g()->nodes()->create<luci::CircleSplitOut>();
    {
      _split_out_1->input(_split);
      _split_out_1->index(0);
      _split_out_1->name("split0");
    }

    _split_out_2 = g()->nodes()->create<luci::CircleSplitOut>();
    {
      _split_out_2->input(_split);
      _split_out_2->index(1);
      _split_out_2->name("split1");
    }

    _add = g()->nodes()->create<luci::CircleAdd>();
    {
      _add->x(_split_out_1);
      _add->y(_split_out_2);
      _add->name("add");
    }
    output()->from(_add);
  }

private:
  luci::CircleSplit *_split = nullptr;
  luci::CircleSplitOut *_split_out_1 = nullptr;
  luci::CircleSplitOut *_split_out_2 = nullptr;
  luci::CircleConst *_split_dim = nullptr;
  luci::CircleAdd *_add = nullptr;
};

} // namespace

TEST(LayerInfoMapTest, simple_test)
{
  SoftmaxTestGraph g;
  g.init();

  luci::LayerInfo info;
  {
    info.name = "test";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  auto map = luci::layer_info_map(g.g(), v);

  EXPECT_EQ("test", map["test"].name);
  EXPECT_EQ(loco::DataType::U8, map["test"].dtype);
  EXPECT_EQ(luci::QuantizationGranularity::ChannelWise, map["test"].granularity);
}

TEST(LayerInfoMapTest, multiple_output_node_test)
{
  SplitAddTestGraph g;
  g.init();

  luci::LayerInfo info;
  {
    info.name = "split0";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  auto map = luci::layer_info_map(g.g(), v);

  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ("split0", map["split0"].name);
  EXPECT_EQ("split1", map["split1"].name);

  EXPECT_EQ(loco::DataType::U8, map["split0"].dtype);
  EXPECT_EQ(luci::QuantizationGranularity::ChannelWise, map["split0"].granularity);
}

TEST(LayerInfoMapTest, invalid_layer_info_multiple_output_node_NEG)
{
  SplitAddTestGraph g;
  g.init();

  luci::LayerInfo info_0;
  {
    info_0.name = "split0";
    info_0.dtype = loco::DataType::U8;
    info_0.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  luci::LayerInfo info_1;
  {
    info_1.name = "split1";
    info_1.dtype = loco::DataType::S16;
    info_1.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info_0);
  v.emplace_back(info_1);

  EXPECT_ANY_THROW(luci::layer_info_map(g.g(), v));
}

TEST(LayerInfoMapTest, duplicate_name_NEG)
{
  SoftmaxTestGraph g;
  g.init();
  g.input()->name("test");

  luci::LayerInfo info;
  {
    info.name = "test";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  EXPECT_ANY_THROW(luci::layer_info_map(g.g(), v));
}

TEST(LayerInfoMapTest, no_name_NEG)
{
  SoftmaxTestGraph g;
  g.init();

  luci::LayerInfo info;
  {
    info.name = "noname";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  EXPECT_ANY_THROW(luci::layer_info_map(g.g(), v));
}
